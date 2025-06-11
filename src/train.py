import os
import sys
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
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from src.utils import set_seed
from models.GPT import GPT
from src.config import TokenizerConfig, ModelConfig, DatasetConfig, TrainConfig


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            train_config: TrainConfig,
            train_loader: DataLoader,
            val_loader: DataLoader,
            device: torch.device,
            master_process: bool
    ):
        self.model = model
        self.train_config = train_config
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
                name=f"{self.train_config.model_name}-{self.train_config.run_name}",
                config=self.train_config.__dict__,
                dir=PROJECT_ROOT
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
                    outputs, loss, _ = self.model(input_ids=input_ids, target_ids=target_ids)
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
                _, val_loss, _ = self.model(input_ids=val_input_ids, target_ids=val_target_ids)
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
    gpt2_small_config = ModelConfig(
        d_embed=768,
        n_layers=12,
        n_heads=12,
        d_head=64,
        d_ff=3072,
        attn_bias=True,
        mlp_bias=True
    )  # 125M
    gpt2_medium_config = ModelConfig(
        d_embed=1024,
        n_layers=24,
        n_heads=16,
        d_head=64,
        d_ff=4096,
        attn_bias=True,
        mlp_bias=True
    )  # 350M
    gpt2_large_config = ModelConfig(
        d_embed=1536,
        n_layers=24,
        n_heads=16,
        d_head=96,
        d_ff=6144,
        attn_bias=True,
        mlp_bias=True
    )  # 760M
    gpt2_xl_config = ModelConfig(
        d_embed=2048,
        n_layers=24,
        n_heads=24,
        d_head=128,
        d_ff=8192,
        attn_bias=True,
        mlp_bias=True
    )  # 1.3B

    ## GPT-2 MoE Configuration
    gpt2_moe_config = ModelConfig(
        n_experts=4,
        n_activated_experts=1
    )  # 294M (125M)
    #gpt2_router_free_moe_config = ModelConfig(
    #    n_experts=4,
    #    n_activated_experts=1,
    #    router_free=True
    #)  # 294M (125M)

    ## nanoGPT Configuration
    nanogpt_config = ModelConfig(
        d_embed=512,
        n_layers=8,
        n_heads=8,
        d_head=64,
        d_ff=2048
    )  # 26M
    nanogpt_moe_config = ModelConfig(
        d_embed=128,
        n_layers=4,
        n_heads=4,
        d_head=32,
        d_ff=512,
        n_experts=4,
        n_activated_experts=1,
    )  # 2.5M (0.9M)

    ## Custom Model Configuration
    pathfinder_config = ModelConfig(
        d_embed=1024,
        n_layers=16,
        n_heads=16,
        d_head=64,
        rank=128,
        d_ff=4096,
        #beta_min=1/2,
        #beta_max=4,
        cross_layer_attention=True
    ) # 117M

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
        print(f"Device: {torch.cuda.get_device_name(device)}")
        master_process = True
        world_size = 1
        rank = 0
        local_rank = 0
        seed_offset = 0
    ## TF32
    torch.set_float32_matmul_precision(train_config.matmul_precision)
    print(f"MatMul Precision: {train_config.matmul_precision}")

    # Reproducibility
    set_seed(42 + seed_offset)

    # Weight&Biases
    if master_process:
        load_dotenv()
        wandb.login(key=os.environ.get("WANDB_API_KEY"))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_config.tokenizer_id,
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer: {tokenizer}")

    # Dataset
    dataset_path = os.path.join(PROJECT_ROOT, dataset_config.local_dir)
    fineweb_dataset = load_from_disk(dataset_path)
    fineweb_dataset.set_format(type="torch", columns=["input_ids"])
    if master_process:
        print(fineweb_dataset)
    fineweb_dataset = fineweb_dataset.train_test_split(test_size=dataset_config.val_size, shuffle=True, seed=train_config.seed)
    if master_process:
        print(f"Train size: {len(fineweb_dataset['train'])}, Test size: {len(fineweb_dataset['test'])}")

    def collate_fn(batch, pad_token_id=tokenizer.pad_token_id):
        """
        custom collate function to pad the input sequences and create target_ids.

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
    elif train_config.model_name == "GPT2-MoE":
        model = GPT(gpt2_moe_config).to(device)
    elif train_config.model_name == "nanoGPT":
        model = GPT(nanogpt_config).to(device)
    elif train_config.model_name == "nanoGPT-MoE":
        model = GPT(nanogpt_moe_config).to(device)
    elif train_config.model_name == "PathFinder":
        model = GPT(pathfinder_config).to(device)
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
        model=model,
        train_config=train_config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        master_process=master_process
    )
    trainer.train()

    # Save model
    if master_process:
        # Save model locally
        output_dir = os.path.join(PROJECT_ROOT, "checkpoints", train_config.model_name, train_config.run_name)
        os.makedirs(output_dir, exist_ok=True)
        try:
            model.save_pretrained(
                output_dir,
                safe_serialization=True
            )
            print(f"Model saved to: {output_dir}")
        except Exception as e:
            print(f"Error saving model: {e}")
        # Push to Hugging Face Hub
        model.push_to_hub(
            repo_id=f"PathFinderKR/{train_config.model_name}-{train_config.run_name}",
            private=True,
            use_auth_token=os.environ.get("HUGGINGFACE_TOKEN")
        )
        print(f"Model pushed to Hugging Face Hub: PathFinderKR/{train_config.model_name}-{train_config.run_name}")

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
# To run DDP training, use:
# torchrun --nproc_per_node=4 train.py