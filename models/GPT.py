import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
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
                torch.tril(
                    torch.ones(config.max_seq_len, config.max_seq_len)
                ).view(1, 1, config.max_seq_len, config.max_seq_len)
            )

    def forward(self, x, kv_cache):
        batch_size, seq_len, _ = x.size()

        # ---------- Linear projection --------------------------------------------------------------------------
        q, k, v = self.qkv_proj(x).split(self.config.d_embed, dim=2)  # [batch_size, seq_len, d_embed]

        # KV cache
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
        kv_seq_len = k.size(1)
        new_kv_cache = (k, v)

        q = q.view(batch_size, seq_len, self.config.n_heads, self.config.d_head).transpose(1, 2)
        k = k.view(batch_size, kv_seq_len, self.config.n_heads, self.config.d_head).transpose(1,2)
        v = v.view(batch_size, kv_seq_len, self.config.n_heads, self.config.d_head).transpose(1,2)
                                                                       # [batch_size, n_heads, seq_len, d_head]

        # ---------- Casual self-attention ----------------------------------------------------------------------
        if self.flash:
            attn = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal=True if seq_len > 1 else False
            )                                                           # [batch_size, n_heads, seq_len, d_head]
        else:
            attn_scores = (q @ k.transpose(-2, -1)) * self.scale        # [batch_size, n_heads, seq_len, seq_len]
            attn_scores = attn_scores.masked_fill(self.mask[:, :, :seq_len, :kv_seq_len] == 0, float('-inf'))
            attn_scores = F.softmax(attn_scores, dim=-1)
            attn_scores = self.attn_dropout(attn_scores)
            attn = attn_scores @ v                                      # [batch_size, n_heads, seq_len, d_head]

        # ---------- Concatenation ------------------------------------------------------------------------------
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.d_embed)
                                                                        # [batch_size, seq_len, d_embed]

        # ---------- Output projection --------------------------------------------------------------------------
        attn_output = self.out_proj(attn)                               # [batch_size, seq_len, d_embed]
        attn_output = self.dropout(attn_output)
        return attn_output, new_kv_cache


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
        x = self.fc1(x)         # [batch_size, seq_len, d_ff]
        x = self.activation(x)
        x = self.fc2(x)         # [batch_size, seq_len, d_embed]
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_embed, eps=config.norm_eps)
        self.attn = MultiHeadAttention(config)
        self.norm2 = nn.LayerNorm(config.d_embed, eps=config.norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, x, kv_cache):
        x = self.norm1(x)
        attn_out, new_kv_cache = self.attn(x, kv_cache=kv_cache)
        x = x + attn_out
        x = self.norm2(x)
        x = x + self.mlp(x)
        return x, new_kv_cache


class GPT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_embed)
        self.positional_encoding = nn.Embedding(config.max_seq_len, config.d_embed)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_embed, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_embed, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    # Kaiming initialization
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

        if kv_cache is None:                                       # ---------- Prefill
            kv_cache = [None] * self.config.n_layers
            start_idx = 0
        else:                                                      # ---------- Decoding
            idx = idx[:, -1:]
            if kv_cache[0][0].size(1) > self.config.max_seq_len -1:
                kv_cache_crop = []
                for k_cache, v_cache in kv_cache:
                    k_cache = k_cache[:, -(self.config.max_seq_len - 1):]
                    v_cache = v_cache[:, -(self.config.max_seq_len - 1):]
                    kv_cache_crop.append((k_cache, v_cache))
                kv_cache = kv_cache_crop
            start_idx = kv_cache[0][0].size(1)

        _, seq_len = idx.size()

        # ---------- Embedding -----------------------------------------------------------------------------------
        tok_embed = self.token_embedding(idx)                                 # [batch_size, seq_len, d_embed]
        pos_idx = torch.arange(start_idx, start_idx + seq_len, device=device) # [seq_len]
        pos_embed = self.positional_encoding(pos_idx).unsqueeze(0)            # [1, seq_len, d_embed]
        x = tok_embed + pos_embed                                             # [batch_size, seq_len, d_embed]
        x = self.dropout(x)

        # ---------- Blocks --------------------------------------------------------------------------------------
        new_kv_cache = []
        for layer_idx, block in enumerate(self.blocks):
            x, kv_cache_layer = block(x, kv_cache=kv_cache[layer_idx])        # [batch_size, seq_len, d_embed]
            new_kv_cache.append(kv_cache_layer)

        # ---------- Final linear layer --------------------------------------------------------------------------
        x = self.norm(x)
        if targets is not None:                                    # ---------- Training
            logits = self.lm_head(x).view(-1, self.config.vocab_size)         # [batch_size * seq_len, vocab_size]
            targets = targets.view(-1)                                        # [batch_size * seq_len]
            loss = F.cross_entropy(logits, targets, ignore_index=-1)
        else:                                                      # ---------- Generation
            logits = self.lm_head(x[:, -1:, :])                               # [batch_size, 1, vocab_size]
            loss = None

        return logits, loss, new_kv_cache

    @torch.inference_mode()
    def generate(
            self,
            idx,
            use_cache: bool,
            max_new_tokens: int,
            temperature: float = 1.0,
            top_k: int = 50,
            tokenizer = None
    ):
        self.eval()
        if not (temperature > 0):
            raise ValueError("temperature must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        kv_cache = None
        for step in range(max_new_tokens):
            # ---------- Truncate --------------------------------------------------------------------------
            idx_input = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]

            # ---------- Forward pass ----------------------------------------------------------------------
            logits, _, kv_cache = self(
                idx_input,
                kv_cache=kv_cache if use_cache else None
            )                                                                 # [batch_size, 1, vocab_size]
            logits = logits[:, -1, :]                                         # [batch_size, vocab_size]

            # ---------- Temperature -----------------------------------------------------------------------
            if temperature != 1.0:
                logits = logits / temperature

            # ---------- Top-k -----------------------------------------------------------------------------
            if top_k is not None:
                k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < k_logits[:, [-1]]] = -float('Inf')

            # ---------- Sample ----------------------------------------------------------------------------
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)                # [batch_size, 1]

            # ---------- Concatenate -----------------------------------------------------------------------
            idx = torch.cat((idx, next_idx), dim=1)                    # [batch_size, seq_len + 1]

            # ---------- Streaming -------------------------------------------------------------------------
            if tokenizer is not None and idx.size(0) == 1:
                try:
                    next_str = tokenizer.decode(next_idx[0].tolist())
                    print(next_str, end='', flush=True)
                except Exception as e:
                    print(f"\nError decoding token: {e}")

        return idx


def main():
    model_config = ModelConfig()
    model = GPT(model_config)
    print(model)
    print(f"Number of parameters: {model.num_params() / 1e6:.2f}M")


if __name__ == "__main__":
    main()