import os
from tqdm import tqdm
from dataclasses import dataclass
from typing import Literal, Optional, Type
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_from_disk
import wandb
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import set_seed
from models.GPT2 import GPT2
from models.GPT2MoE import GPT2MoE
from models.PathFinderV1 import PathFinder


@dataclass
class TokenizerConfig:
    tokenizer_id: Literal["gpt2"] = "gpt2"

@dataclass
class ModelConfig:
    vocab_size: int = 50304  # 50000 BPE merges + 256 bytes + 1 <|endoftext|> = 50257 -> 50304 for GPU efficiency
    max_seq_len: int = 1024
    d_embed: int = 768
    n_layers: int = 12
    norm_eps: float = 1e-5
    dropout: float = 0.1

    # Attention
    attn_type: Literal["mha", "gqa", "mla"] = "mha"
    n_heads: int = 12
    d_head: int = d_embed // n_heads
    attn_bias: bool = False
    n_kv_heads: Optional[int] = None
    d_latent: Optional[int] = None
    ## Mixture of Attention Heads
    moh: bool = False
    n_activated_heads: Optional[int] = None
    n_shared_heads: Optional[int] = None

    # FeedForward
    d_ff: int = d_embed * 4
    mlp_bias: bool = False
    activation: Type[nn.Module] = nn.GELU
    d_ff_multiplier: Optional[float] = None
    d_ff_multiple_of: int = 256
    ## Mixture of Experts
    moe: bool = False
    n_experts: Optional[int] = None
    n_activated_experts: Optional[int] = None
    n_shared_experts: Optional[int] = None

@dataclass
class DatasetConfig:
    local_dir: str = f"datasets/FineWeb-Edu/10B"
    val_size: float = 0.01

@dataclass
class TrainConfig:
    debug: bool = False
    wandb_project: str = "PathFinder"
    model_name: Literal["GPT2-small", "GPT2-medium", "GPT2-large", "GPT2-xl", "GPT2-xs", "GPT2-MoE", "PathFinder"] = "GPT2-xs"
    run_name = f"{model_name}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_dir: str = f"checkpoints/{model_name}"
    num_workers: int = 4

    # Training
    per_device_train_batch_size: int = 128
    per_device_eval_batch_size: int = 256
    num_train_epochs: int = 1
    learning_rate: float = 2e-3
    weight_decay: float = 0.1
    optim: torch.optim.Optimizer = torch.optim.AdamW
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 512 // per_device_train_batch_size
    eval_steps: int = 1000
    seed: int = 42
    ## Precision
    mixed_precision: bool = True
    matmul_precision: Literal["highest", "high", "medium"] = "high"


class Trainer:
    def __init__(
            self,
            train_config: TrainConfig,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            device: str,
            master_process: bool
    ):
        self.train_config = train_config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.master_process = master_process

    def train(self):
        if self.master_process:
            wandb.init(
                project=self.train_config.wandb_project,
                name=self.train_config.run_name,
                config=self.train_config.__dict__
            )
            wandb.watch(self.model, log="all")

        total_steps = (len(self.train_loader) * self.train_config.num_train_epochs // self.train_config.gradient_accumulation_steps)
        warmup_steps = int(self.train_config.warmup_ratio * total_steps)

        optimizer = self.train_config.optim(
            self.model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
            betas=self.train_config.betas,
            eps=self.train_config.eps,
            fused=True
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        progress_bar = tqdm(total=total_steps, desc="Training", disable=not self.master_process)
        step = 0

        for epoch in range(self.train_config.num_train_epochs):
            self.train_loader.sampler.set_epoch(epoch)
            for batch_idx, batch in enumerate(self.train_loader):
                self.model.train()
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)

                if self.train_config.mixed_precision:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs, loss = self.model(input_ids, target_ids)
                else:
                    outputs, loss = self.model(input_ids, target_ids)
                loss = loss / self.train_config.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.train_config.gradient_accumulation_steps == 0:
                    grad_norm = clip_grad_norm_(self.model.parameters(), self.train_config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    step += 1

                    if self.master_process:
                        wandb.log({
                            "Train Loss": loss.item() * self.train_config.gradient_accumulation_steps,
                            "Learning Rate": scheduler.get_last_lr()[0],
                            "Grad Norm": grad_norm,
                            "Epoch": epoch + 1
                        })
                        progress_bar.set_postfix(
                            loss=f"{loss.item() * self.train_config.gradient_accumulation_steps:.4f}",
                            lr=f"{scheduler.get_last_lr()[0]:.6f}",
                            grad_norm=f"{grad_norm:.4f}",
                            epoch=epoch + 1
                        )
                        progress_bar.update(1)

                    if step % self.train_config.eval_steps == 0:
                        self.validate()

        self.model.eval()
        self.validate()  # Final validation
        if self.master_process:
            progress_bar.close()
            wandb.finish()

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_val_loss = 0.0
        total_samples = 0

        for val_batch in self.val_loader:
            val_input_ids = val_batch["input_ids"].to(self.device)
            val_target_ids = val_batch["target_ids"].to(self.device)
            if self.train_config.mixed_precision:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, val_loss = self.model(val_input_ids, val_target_ids)
            else:
                _, val_loss = self.model(val_input_ids, val_target_ids)
            total_val_loss += val_loss.item() * val_input_ids.size(0)
            total_samples += val_input_ids.size(0)

        if self.master_process:
            wandb.log({"Val Loss": total_val_loss / total_samples})


def main():
    # Reproducibility
    set_seed(42)

    # Configuration
    tokenizer_config = TokenizerConfig()
    #model_config = ModelConfig()
    dataset_config = DatasetConfig()
    train_config = TrainConfig()

    # GPT-2 Configuration
    gpt2_small_config = ModelConfig(
        vocab_size=50304,
        max_seq_len=1024,
        d_embed=768,
        n_layers=12,
        norm_eps=1e-5,
        dropout=0.1,
        # Attention
        n_heads=12,
        d_head=64,
        attn_bias=True,
        # FeedForward
        d_ff=3072,
        mlp_bias=True,
        activation=nn.GELU
    )  # 124M

    gpt2_medium_config = ModelConfig(
        vocab_size=50304,
        max_seq_len=1024,
        d_embed=1024,
        n_layers=24,
        norm_eps=1e-5,
        dropout=0.1,
        # Attention
        n_heads=16,
        d_head=64,
        attn_bias=True,
        # FeedForward
        d_ff=4096,
        mlp_bias=True,
        activation=nn.GELU
    )  # 350M

    gpt2_large_config = ModelConfig(
        vocab_size=50304,
        max_seq_len=1024,
        d_embed=1536,
        n_layers=24,
        norm_eps=1e-5,
        dropout=0.1,
        # Attention
        n_heads=16,
        d_head=96,
        attn_bias=True,
        # FeedForward
        d_ff=6144,
        mlp_bias=True,
        activation=nn.GELU
    )  # 760M

    gpt2_xl_config = ModelConfig(
        vocab_size=50304,
        max_seq_len=1024,
        d_embed=2048,
        n_layers=24,
        norm_eps=1e-5,
        dropout=0.1,
        # Attention
        n_heads=24,
        d_head=128,
        attn_bias=True,
        # FeedForward
        d_ff=8192,
        mlp_bias=True,
        activation=nn.GELU
    )  # 1.3B

    gpt2_xs_config = ModelConfig(
        vocab_size=50304,
        max_seq_len=128,
        d_embed=256,
        n_layers=8,
        norm_eps=1e-5,
        dropout=0.1,
        # Attention
        n_heads=8,
        d_head=32,
        attn_bias=True,
        # FeedForward
        d_ff=1024,
        mlp_bias=True,
        activation=nn.GELU
    )  # 19M

    gpt2_moe_config = ModelConfig(
        vocab_size=50304,
        max_seq_len=1024,
        d_embed=768,
        n_layers=12,
        norm_eps=1e-5,
        dropout=0.1,
        # Attention
        n_heads=12,
        d_head=64,
        attn_bias=True,
        # FeedForward
        d_ff=3072,
        mlp_bias=True,
        activation=nn.GELU,
        ## Mixture of Experts
        n_experts=4,
        n_activated_experts=1
    )  # 294M

    # Device
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(ddp_local_rank)
    master_process = ddp_rank == 0
    torch.set_float32_matmul_precision(train_config.matmul_precision)  # Tensor Cores

    # Weight&Biases
    if master_process:
        wandb.login(key=os.environ.get("WANDB_API_KEY"))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.tokenizer_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    fineweb_dataset = load_from_disk(dataset_config.local_dir)
    fineweb_dataset.set_format(type="torch", columns=["input_ids"])
    print(fineweb_dataset)
    fineweb_dataset = fineweb_dataset.train_test_split(test_size=dataset_config.val_size, shuffle=True, seed=train_config.seed)

    def collate_fn(batch, pad_token_id=tokenizer.pad_token_id):
        """
        Custom collate function to pad the input sequences and create target_ids.

        Args:
            batch (list): List of dictionaries containing input IDs.
            pad_token_id (int): Padding token ID.

        Returns:
            dict: Dictionary containing padded input IDs and target_ids.
        """
        input_ids = pad_sequence([example["input_ids"] for example in batch], batch_first=True, padding_value=pad_token_id)
        target_ids = input_ids.clone()
        target_ids[:, :-1] = input_ids[:, 1:]
        target_ids[:, -1] = pad_token_id
        target_ids[target_ids == pad_token_id] = -1
        return {"input_ids": input_ids, "target_ids": target_ids}

    train_sampler = DistributedSampler(fineweb_dataset["train"], num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True)
    val_sampler = DistributedSampler(fineweb_dataset["test"], num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False)

    train_loader = DataLoader(
        fineweb_dataset["train"],
        sampler=train_sampler,
        collate_fn=collate_fn,
        batch_size=train_config.per_device_train_batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        fineweb_dataset["test"],
        sampler=val_sampler,
        collate_fn=collate_fn,
        batch_size=train_config.per_device_eval_batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=True
    )

    # Model
    if train_config.model_name == "GPT2-small":
        model = GPT2(gpt2_small_config).to(device)
    elif train_config.model_name == "GPT2-medium":
        model = GPT2(gpt2_medium_config).to(device)
    elif train_config.model_name == "GPT2-large":
        model = GPT2(gpt2_large_config).to(device)
    elif train_config.model_name == "GPT2-xl":
        model = GPT2(gpt2_xl_config).to(device)
    elif train_config.model_name == "GPT2-xs":
        model = GPT2(gpt2_xs_config).to(device)
    elif train_config.model_name == "GPT2-MoE":
        model = GPT2MoE(gpt2_moe_config).to(device)
    # elif train_config.model_name == "PathFinder":
    #    model = PathFinder().to(device)
    else:
        raise ValueError(f"Unknown model name: {train_config.model_name}")
    model = DDP(model, device_ids=[ddp_local_rank])
    if master_process:
        print(model)

    # Training
    trainer = Trainer(
        train_config=train_config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        master_process=master_process
    )
    trainer.train()

    destroy_process_group()

    # Save model
    if master_process:
        os.makedirs(train_config.output_dir, exist_ok=True)
        torch.save(model.module.state_dict(), os.path.join(train_config.output_dir, f"{train_config.run_name}.pt"))
        print(f"Model saved to {os.path.join(train_config.output_dir, f'{train_config.run_name}.pt')}")


if __name__ == "__main__":
    main()