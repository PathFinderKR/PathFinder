import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
#from transformer_engine.pytorch import fp8_autocast
from src.config import ModelConfig


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
            self.scale = config.d_head ** -0.5
            self.attn_dropout = nn.Dropout(config.dropout)
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(1, 1, config.max_seq_len, config.max_seq_len)
            )

    def forward(self, x, kv_cache: tuple = None):
        batch_size, seq_len, _ = x.size()

        # Linear projection
        q, k, v = self.qkv_proj(x).split(self.config.d_embed, dim=2)  # [batch_size, seq_len, d_embed]
        q = q.view(batch_size, seq_len, self.config.n_heads, self.config.d_head).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_head]
        k = k.view(batch_size, seq_len, self.config.n_heads, self.config.d_head).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_head]
        v = v.view(batch_size, seq_len, self.config.n_heads, self.config.d_head).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_head]

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat((k_cache, k), dim=2)  # [batch_size, n_heads, seq_len + cache_len, d_head]
            v = torch.cat((v_cache, v), dim=2)  # [batch_size, n_heads, seq_len + cache_len, d_head]
        kv_current = (k, v)

        # Casual self-attention
        if self.flash:
            attn = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal=True
            )  # [batch_size, n_heads, seq_len, d_head]
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [batch_size, n_heads, seq_len, seq_len]
            attn = attn.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            attn = attn @ v  # [batch_size, n_heads, seq_len, d_head]

        # Reshape and concatenate heads
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.d_embed)  # [batch_size, seq_len, d_embed]

        # Output projection
        attn = self.out_proj(attn)  # [batch_size, seq_len, d_embed]
        attn = self.dropout(attn)

        return attn


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_embed, config.d_ff, bias=config.mlp_bias)
        self.activation = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
        }[config.activation]()
        self.fc2 = nn.Linear(config.d_ff, config.d_embed, bias=config.mlp_bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)  # [batch_size, seq_len, d_ff]
        x = self.activation(x)
        x = self.fc2(x)  # [batch_size, seq_len, d_embed]
        x = self.dropout(x)
        return x


class Expert(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_embed, config.d_ff, bias=config.mlp_bias)
        self.activation = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
        }[config.activation]()
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

    def forward(self, x):
        logits = self.gate(x)  # [batch_size, seq_len, n_experts]
        scores = F.softmax(logits, dim=-1)
        scores, indices = torch.topk(scores, self.top_k, dim=-1)  # [batch_size, seq_len, top_k]
        return scores, indices


class MoE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.top_k = config.n_activated_experts
        if config.n_experts is None or config.n_activated_experts is None:
            raise ValueError("n_experts and n_activated_experts must be specified for MoE")
        if config.n_experts < config.n_activated_experts:
            raise ValueError("n_experts must be greater than or equal to n_activated_experts")
        self.router = Router(config)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.n_experts)])
        if config.n_shared_experts is not None:
            self.shared_experts = nn.ModuleList([Expert(config) for _ in range(config.n_shared_experts)])
        else:
            self.shared_experts = None

    def forward(self, x):
        batch_size, seq_len, d_embed = x.size()
        scores, indices = self.router(x)

        x_flat = x.view(-1, d_embed)  # [batch * seq, d_embed]
        scores_flat = scores.view(-1, self.top_k)  # [batch * seq, top_k]
        indices_flat = indices.view(-1, self.top_k)  # [batch * seq, top_k]

        y_flat = torch.zeros_like(x_flat)

        for expert_idx, expert in enumerate(self.experts):
            mask = (indices_flat == expert_idx)
            selected_pos = mask.any(dim=-1)  # [batch * seq]

            if selected_pos.any():
                expert_input = x_flat[selected_pos]  # [top_k, d_embed]
                expert_output = expert(expert_input)

                selected_mask_idx, selected_topk_idx = torch.where(mask)
                gating_scores = scores_flat[selected_mask_idx, selected_topk_idx].unsqueeze(1)  # [top_k, 1]

                weighted_output = expert_output * gating_scores  # [top_k, d_embed]
                y_flat[selected_pos] += weighted_output  # [batch * seq, d_embed]

        if self.shared_experts is not None:
            shared_output = sum([expert(x_flat) for expert in self.shared_experts]) / len(self.shared_experts)
            y_flat += shared_output  # [batch * seq, d_embed]

        y = y_flat.view(batch_size, seq_len, d_embed)
        return y


class RouterFreeMoE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.n_experts)])

    def forward(self, x):
        batch_size, seq_len, d_embed = x.size()
        x_flat = x.view(-1, d_embed)  # [batch * seq, d_embed]

        # Step 1: Expert selection
        deltas = []
        with torch.no_grad():
            # if Device compute capabilities is 8.9 or higher, use fp8_autocast
            if torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.get_device_capability()[1] >= 9:
                #ctx = fp8_autocast(enabled=True)
                ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            else:
                ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            with ctx:
                for expert in self.experts:
                    expert_output = expert(x_flat)  # [batch * seq, d_embed]
                    delta = (expert_output - x_flat).norm(dim=-1)  # [batch * seq]
                    deltas.append(delta)
            deltas = torch.stack(deltas, dim=0)  # [n_experts, batch * seq]
            best_expert_idx = deltas.argmax(dim=0)  # [batch * seq]

        # Step 2: Apply the best expert
        y_flat = torch.zeros_like(x_flat)
        for expert_idx, expert in enumerate(self.experts):
            mask = (best_expert_idx == expert_idx)  # [batch * seq]
            if mask.any():
                selected_input = x_flat[mask]  # [num_selected, d_embed]
                selected_output = expert(selected_input)  # [num_selected, d_embed]
                y_flat[mask] = selected_output  # [batch * seq, d_embed]
        y = y_flat.view(batch_size, seq_len, d_embed)
        return y


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_embed, eps=config.norm_eps)
        self.attn = MultiHeadAttention(config)
        self.norm2 = nn.LayerNorm(config.d_embed, eps=config.norm_eps)
        if config.n_experts is not None:
            if config.router_free:
                self.mlp = RouterFreeMoE(config)
            else:
                self.mlp = MoE(config)
        else:
            self.mlp = FeedForward(config)

    def forward(self, x, kv_cache = None):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module, PyTorchModelHubMixin):
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

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) - self.lm_head.weight.numel()

    def num_active_params(self):
        pass

    def forward(self, idx, targets=None, kv_cache=None):
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
    model_config = ModelConfig()
    model = GPT(model_config)
    print(model)
    print(f"Number of parameters: {model.num_params() / 1e6:.2f}M")


if __name__ == "__main__":
    main()