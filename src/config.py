from dataclasses import dataclass
from typing import Literal, Optional
from datetime import datetime
import torch


@dataclass
class TrainConfig:
    debug: bool = False
    wandb_project: str = "GPT"
    model_name: Literal[
        "GPT2-small", "GPT2-medium", "GPT2-large", "GPT2-xl",  # GPT-2
        "GPT2-MoE", "PathFinder",                              # custom models
        "nanoGPT", "nanoGPT-MoE"                               # nano versions
    ] = "GPT2-small"
    run_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # Training
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 512 // per_device_train_batch_size  # 512 = global batch size
    num_train_epochs: int = 1
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    optim: torch.optim.Optimizer = torch.optim.AdamW
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    eval_steps: int = 5000
    seed: int = 42
    ## Precision
    mixed_precision: bool = True
    matmul_precision: Literal["highest", "high", "medium"] = "high"

@dataclass
class DatasetConfig:
    dataset_id: Literal["HuggingFaceFW/fineweb-edu"] = "HuggingFaceFW/fineweb-edu"
    remote_name: Optional[str] = "sample-10BT"
    split: Optional[str] = "train"
    local_dir: str = f"datasets/FineWeb-Edu/10B"
    val_size: float = 0.01

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
    n_heads: int = 12
    d_head: int = 64
    flash: bool = False
    attn_bias: bool = False
    attn_type: Literal["mha", "gqa", "mla"] = "mha"
    n_kv_heads: Optional[int] = None
    rank: Optional[int] = None
    ## Mixture of Attention Heads
    n_activated_heads: Optional[int] = None
    n_shared_heads: Optional[int] = None

    # FeedForward
    d_ff: int = 3072
    mlp_bias: bool = False
    activation: Literal["relu", "gelu"] = "gelu"
    d_ff_multiplier: Optional[float] = None
    d_ff_multiple_of: int = 256
    ## Mixture of Experts
    router_free: bool = False
    n_experts: Optional[int] = None
    n_activated_experts: Optional[int] = None
    n_shared_experts: Optional[int] = None

@dataclass
class GenerationConfig:
    checkpoint_path: str = "checkpoints/nanoGPT/2025-05-31_17-46-15"
    matmul_precision: Literal["highest", "high", "medium"] = "high"
    use_cache: bool = True
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 50