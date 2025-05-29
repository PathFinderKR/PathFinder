import os
from dataclasses import dataclass
from typing import Literal, Optional, Type
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import set_seed
from models.GPT2 import GPT2


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
class GenerationConfig:
    checkpoint_path: str = "checkpoints/GPT2/GPT2-2025-05-22_23-04-09.pt"
    dtype: torch.dtype = torch.int8
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 50


def main():
    # Reproducibility
    set_seed(42)

    # Configuration
    tokenizer_config = TokenizerConfig()
    model_config = ModelConfig()
    generation_config = GenerationConfig()

    # Device
    device = torch.device("cuda")
    torch.set_float32_matmul_precision('high')  # Tensor Cores

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.tokenizer_id)

    # Model
    model = GPT2(model_config).to(device=device)
    #model = model.compile(model)
    #if os.path.exists(generation_config.checkpoint_path):
    #    checkpoint = torch.load(generation_config.checkpoint_path, map_location=device)
    #    model.load_state_dict(checkpoint['model_state_dict'])
    #else:
    #    raise FileNotFoundError(f"No checkpoint found at {generation_config.checkpoint_path}")

    # Generate
    while True:
        print("=" * 50)
        print("User prompt: ")
        user_prompt = input("> ")
        if user_prompt.lower() == "exit":
            break
        input_ids = tokenizer.encode(user_prompt, return_tensors="pt").to(device)
        print("-" * 50)
        print("🤖 Model Response:")
        output = model.generate(
            input_ids,
            max_new_tokens=generation_config.max_new_tokens,
            temperature=generation_config.temperature,
            top_k=generation_config.top_k
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(response)


if __name__ == "__main__":
    main()