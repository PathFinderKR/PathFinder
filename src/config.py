from dataclasses import dataclass, field
from typing import Literal, Optional
from datetime import datetime
import torch
from transformers import PretrainedConfig

#####
# 1. Attention Weight Decay
# 2. Kaiming He init 0.2 -> 0.1

@dataclass
class TrainConfig:
    debug: bool = False
    wandb_project: str = "Test"
    model_name: str = "GPT2-small"
    run_name: str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # Training
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 512 // per_device_train_batch_size  # 512 = global batch size
    num_train_epochs: int = 1
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    attn_decay: float = 0.1
    optim: torch.optim.Optimizer = torch.optim.AdamW
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    eval_steps: int = 5000
    seed: int = 42
    ## Precision
    mixed_precision: bool = True
    matmul_precision: Literal["highest", "high", "medium"] = "medium"


@dataclass
class ModelConfig(PretrainedConfig):
    model_type = "GPT-2"
    vocab_size: int = 50304  # 50000 BPE merges + 256 bytes + 1 <|endoftext|> = 50257 -> 50304 for GPU efficiency
    max_seq_len: int = 1024
    d_embed: int = 768
    n_layers: int = 12

    # Attention
    attn_type: Literal["MHA", "GQA", "MLA"] = "MHA"
    flash: bool = True
    flash_decode: bool = False
    n_heads: int = 12
    d_head: int = 64
    attn_bias: bool = False
    n_kv_heads: Optional[int] = None
    rank: Optional[int] = None

    # FeedForward
    d_ff: int = 3072
    mlp_bias: bool = False
    activation: Literal["relu", "gelu"] = "gelu"
    d_ff_multiplier: Optional[float] = None
    d_ff_multiple_of: int = 256
    ## Layer-wise scaling
    beta_min: Optional[float] = None
    beta_max: Optional[float] = None
    ## Mixture of Experts
    n_experts: Optional[int] = None
    n_activated_experts: Optional[int] = None
    n_shared_experts: Optional[int] = None

    attn_std: float = 0.01
    ff_std: float = 0.02
    embed_std: float = 0.02
    norm_eps: float = 1e-5
    dropout: float = 0.01


## GPT-2 Configuration
gpt2_small_config = ModelConfig(
    d_embed=768,
    n_layers=12,
    n_heads=12,
    d_head=64,
    d_ff=3072,
    attn_bias=False,
    mlp_bias=True
)  # 124.48M
gpt2_medium_config = ModelConfig(
    d_embed=1024,
    n_layers=24,
    n_heads=16,
    d_head=64,
    d_ff=4096,
    attn_bias=False,
    mlp_bias=True
)  # 354.87M
gpt2_large_config = ModelConfig(
    d_embed=1536,
    n_layers=24,
    n_heads=16,
    d_head=96,
    d_ff=6144,
    attn_bias=False,
    mlp_bias=True
)  # 758.80M
gpt2_xl_config = ModelConfig(
    d_embed=2048,
    n_layers=24,
    n_heads=24,
    d_head=128,
    d_ff=8192,
    attn_bias=False,
    mlp_bias=True
)  # 1.3B

## nanoGPT Configuration
nanogpt_config = ModelConfig(
    d_embed=512,
    n_layers=8,
    n_heads=8,
    d_head=64,
    d_ff=2048
)  # 26M

## Custom Model Configuration
mla_config = ModelConfig(
    d_embed=768,
    n_layers=12,
    n_heads=12,
    d_head=64,
    d_ff=3072,
    attn_type="MLA",
    rank=32,
    attn_bias=False,
    mlp_bias=True
)  # 111.17M

model_config = gpt2_small_config


@dataclass
class TokenizerConfig:
    tokenizer_id: Literal["gpt2"] = "gpt2"


@dataclass
class DatasetConfig:
    dataset_id: Literal["HuggingFaceFW/fineweb-edu"] = "HuggingFaceFW/fineweb-edu"
    remote_name: Optional[str] = "sample-10BT"
    split: Optional[str] = "train"
    local_dir: str = f"datasets/FineWeb-Edu/10B"
    val_size: float = 0.01


@dataclass
class GenerationConfig:
    checkpoint_path: str = "checkpoints/PathFinder/2025-06-09_21-14-00"
    model_id: str = "GPT2-small-2025-06-09_21-14-00"
    matmul_precision: Literal["highest", "high", "medium"] = "high"
    seed: int = 101
    use_cache: bool = True
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 50
    speedometer: bool = False