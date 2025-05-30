import os
from dotenv import load_dotenv
from tqdm import tqdm
from contextlib import nullcontext
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
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import set_seed
from models.GPT2 import GPT
from src.config import TokenizerConfig, ModelConfig, DatasetConfig, TrainConfig


class Trainer:
    def __init__(
            self,
            train_config: TrainConfig,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            device: torch.device,
            master_process: bool
    ):
        self.train_config = train_config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.master_process = master_process
        if self.train_config.mixed_precision:
            self.ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            self.ctx = nullcontext()

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
            for batch_idx, batch in enumerate(self.train_loader):
                self.model.train()
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)

                with self.ctx:
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
                            "Grad Norm": grad_norm
                        })
                        progress_bar.set_postfix(
                            loss = f"{loss.item() * self.train_config.gradient_accumulation_steps:.4f}",
                            lr = f"{scheduler.get_last_lr()[0]:.6f}",
                            grad_norm = f"{grad_norm:.4f}",
                            epoch = epoch + 1,
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

            with self.ctx:
                _, val_loss = self.model(val_input_ids, val_target_ids)
            total_val_loss += val_loss.item() * val_input_ids.size(0)
            total_samples += val_input_ids.size(0)

        avg_val_loss = total_val_loss / total_samples
        val_perplexity = torch.exp(torch.tensor(avg_val_loss)).item()

        if self.master_process:
            wandb.log({
                "Val Loss": total_val_loss / total_samples,
                "Val Perplexity": val_perplexity
            })


def main():
    # Configuration
    tokenizer_config = TokenizerConfig()
    #model_config = ModelConfig()
    dataset_config = DatasetConfig()
    train_config = TrainConfig()

    ## GPT-2 Configuration
    gpt2_small_config = ModelConfig()  # 124M
    gpt2_medium_config = ModelConfig(
        d_embed=1024,
        n_layers=24,
        n_heads=16,
        d_head=64,
        d_ff=4096
    )  # 350M
    gpt2_large_config = ModelConfig(
        d_embed=1536,
        n_layers=24,
        n_heads=16,
        d_head=96,
        d_ff=6144
    )  # 760M
    gpt2_xl_config = ModelConfig(
        d_embed=2048,
        n_layers=24,
        n_heads=24,
        d_head=128,
        d_ff=8192,
    )  # 1.3B
    gpt2_xs_config = ModelConfig(
        max_seq_len=128,
        d_embed=256,
        n_layers=8,
        n_heads=8,
        d_head=32,
        d_ff=1024,
    )  # 19M
    gpt2_moe_config = ModelConfig(
        n_experts=4,
        n_activated_experts=1
    )  # 294M
    gpt2_router_free_moe_config = ModelConfig(
        n_experts=4,
        n_activated_experts=1,
        router_free=True
    )  # 294M

    # Device
    ## Distributed Data Parallel (DDP) setup
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        world_size = int(os.environ.get('WORLD_SIZE'))
        rank = int(os.environ.get('RANK'))
        local_rank = int(os.environ.get('LOCAL_RANK'))
        master_process = rank == 0
        seed_offset = rank
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
        print(f"Using DDP on device: {device}")
        assert train_config.gradient_accumulation_steps % world_size == 0, \
            "Gradient accumulation steps must be divisible by world size for DDP."
        train_config.gradient_accumulation_steps //= world_size  # Update global batch size for DDP
    else:
        device = torch.device("cuda")
        master_process = True
        world_size = 1
        rank = 0
        local_rank = 0
        seed_offset = 0
    ## TF32
    torch.set_float32_matmul_precision(train_config.matmul_precision)

    # Reproducibility
    set_seed(42 + seed_offset)

    # Weight&Biases
    if master_process:
        load_dotenv()
        wandb.login(key=os.environ.get("WANDB_API_KEY"))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.tokenizer_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    if train_config.model_name == "GPT2-xs":
        fineweb_dataset = load_from_disk("datasets/FineWeb-Edu/10B-128")
    else:
        fineweb_dataset = load_from_disk(dataset_config.local_dir)
    fineweb_dataset.set_format(type="torch", columns=["input_ids"])
    if master_process:
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

    if ddp:
        train_sampler = DistributedSampler(fineweb_dataset["train"], num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(fineweb_dataset["test"], num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        fineweb_dataset["train"],
        collate_fn=collate_fn,
        batch_size=train_config.per_device_train_batch_size,
        sampler=train_sampler,
        shuffle=False if ddp else True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        fineweb_dataset["test"],
        collate_fn=collate_fn,
        batch_size=train_config.per_device_eval_batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Model
    if train_config.model_name == "GPT2-small":
        model = GPT(gpt2_small_config).to(device)
    elif train_config.model_name == "GPT2-medium":
        model = GPT(gpt2_medium_config).to(device)
    elif train_config.model_name == "GPT2-large":
        model = GPT(gpt2_large_config).to(device)
    elif train_config.model_name == "GPT2-xl":
        model = GPT(gpt2_xl_config).to(device)
    elif train_config.model_name == "GPT2-xs":
        model = GPT(gpt2_xs_config).to(device)
    elif train_config.model_name == "GPT2-MoE":
        model = GPT(gpt2_moe_config).to(device)
    elif train_config.model_name == "PathFinder":
        model = GPT(gpt2_router_free_moe_config).to(device)
    else:
        raise ValueError(f"Unknown model name: {train_config.model_name}")
    model = torch.compile(model)
    if ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )

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

    if ddp:
        destroy_process_group()

    # Save model
    if master_process:
        os.makedirs(train_config.output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(train_config.output_dir, f"{train_config.run_name}.pt"))
        print(f"Model saved to {os.path.join(train_config.output_dir, f'{train_config.run_name}.pt')}")


if __name__ == "__main__":
    main()
# To run DDP training, use:
# torchrun --nproc_per_node=4 src/train.py