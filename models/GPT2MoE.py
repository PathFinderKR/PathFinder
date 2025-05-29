import math
from dataclasses import dataclass
from typing import Optional, Type
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    d_head: int = d_embed // n_heads
    attn_bias: bool = True

    # FeedForward
    d_ff: int = d_embed * 4
    mlp_bias: bool = True
    activation: Type[nn.Module] = nn.GELU
    ## Mixture of Experts
    n_experts: Optional[int] = 4
    n_activated_experts: Optional[int] = 1


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_embed % config.n_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.config = config
        self.qkv_proj = nn.Linear(config.d_embed, 3 * config.d_embed, bias=config.attn_bias)
        self.out_proj = nn.Linear(config.d_embed, config.d_embed, bias=config.attn_bias)
        self.dropout = nn.Dropout(config.dropout)

        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            print("Flash attention not available, using standard implementation.")
            self.scale = config.d_head ** -0.5
            self.attn_dropout = nn.Dropout(config.dropout)
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(1, 1, config.max_seq_len, config.max_seq_len)
            )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Linear projection
        q, k, v = self.qkv_proj(x).split(self.config.d_embed, dim=2)  # [batch_size, seq_len, d_embed]
        q = q.view(batch_size, seq_len, self.config.n_heads, self.config.d_head).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_head]
        k = k.view(batch_size, seq_len, self.config.n_heads, self.config.d_head).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_head]
        v = v.view(batch_size, seq_len, self.config.n_heads, self.config.d_head).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_head]

        # Casual self-attention
        if self.flash:
            attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.config.dropout if self.training else 0.0, is_causal=True)  # [batch_size, n_heads, seq_len, d_head]
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [batch_size, n_heads, seq_len, seq_len]
            attn = attn.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            attn = attn @ v  # [batch_size, n_heads, seq_len, d_head]
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.d_embed)  # [batch_size, seq_len, d_embed]

        # Output projection
        attn = self.out_proj(attn)  # [batch_size, seq_len, d_embed]
        attn = self.dropout(attn)

        return attn


class Expert(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_embed, config.d_ff, bias=config.mlp_bias)
        self.activation = config.activation()
        self.fc2 = nn.Linear(config.d_ff, config.d_embed, bias=config.mlp_bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)  # [batch_size, seq_len, d_ff]
        x = self.activation(x)
        x = self.fc2(x)  # [batch_size, seq_len, d_embed]
        x = self.dropout(x)
        return x


class Router(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.top_k = config.n_activated_experts
        self.gate = nn.Linear(config.d_embed, config.n_experts)
        self.noise_linear = nn.Linear(config.d_embed, config.n_experts)

    def forward(self, x):
        logits = self.gate(x)
        noise_std = F.softplus(self.noise_linear(x))
        noise = torch.randn_like(logits) * noise_std
        noisy_logits = logits + noise
        topk_vals, topk_idx = noisy_logits.topk(self.top_k, dim=-1)

        mask = torch.full_like(noisy_logits, float("-inf"))
        sparse_logits = mask.scatter(-1, topk_idx, topk_vals)
        probs = F.softmax(sparse_logits, dim=-1)
        return probs, topk_idx


class MoE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.router = Router(config)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.n_experts)])

    def forward(self, x):
        gating_probs, indices = self.router(x)
        batch, seq_len, _ = x.size()

        final = torch.zeros_like(x)
        flat_x = x.view(-1, x.size(-1))
        flat_probs = gating_probs.view(-1, gating_probs.size(-1))
        flat_idx = indices.view(-1, indices.size(-1))

        for i, expert in enumerate(self.experts):
            mask = (flat_idx == i).any(dim=-1)
            if not mask.any():
                continue
            expert_in = flat_x[mask]
            expert_out = expert(expert_in)
            scores = flat_probs[mask, i].unsqueeze(1)
            weighted = expert_out * scores
            final.view(-1, final.size(-1))[mask] += weighted

        return final


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_embed, eps=config.norm_eps)
        self.attn = MultiHeadAttention(config)
        self.norm2 = nn.LayerNorm(config.d_embed, eps=config.norm_eps)
        self.mlp = MoE(config)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(module.bias, -bound, bound)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))


class GPT2MoE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_embed)
        self.positional_encoding = nn.Embedding(config.max_seq_len, config.d_embed)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_embed, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

        self.apply(_init_weights)

    def num_params(self):
        unique = {p.data_ptr(): p for p in self.parameters()}
        return sum(p.numel() for p in unique.values())

    def forward(self, idx, targets=None):
        device = idx.device
        batch_size, seq_len = idx.size()

        # Embedding
        tok_embed = self.embedding(idx)  # [batch_size, seq_len, d_embed]
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)  # [seq_len]
        pos_embed = self.positional_encoding(pos)  # [seq_len, d_embed]
        x = tok_embed + pos_embed  # [batch_size, seq_len, d_embed]
        x = self.dropout(x)

        # Blocks
        for block in self.blocks:
            x = block(x)  # [batch_size, seq_len, d_embed]

        # Final normalization and linear layer
        x = self.norm(x)
        if targets is not None:
            logits = self.lm_head(x)  # [batch_size, seq_len, vocab_size]
            logits = logits.view(-1, self.config.vocab_size)  # [batch_size * seq_len, vocab_size]
            targets = targets.view(-1)  # [batch_size * seq_len]
            loss = F.cross_entropy(logits, targets, ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])  # [batch_size, 1, vocab_size]
            loss = None

        return logits, loss

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int = 50):
        self.eval()
        if not (temperature > 0):
            raise ValueError("temperature must be positive")

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)  # [batch_size, 1, vocab_size]
            logits = logits[:, -1, :] / temperature  # [batch_size, vocab_size]
            if top_k is not None:
                k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < k_logits[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
            idx = torch.cat((idx, next_idx), dim=1)  # [batch_size, seq_len + 1]
        return idx


def main():
    config = ModelConfig()
    model = GPT2MoE(config)
    print(f"Number of parameters: {model.num_params() / 1e6:.2f}M")


if __name__ == "__main__":
    main()