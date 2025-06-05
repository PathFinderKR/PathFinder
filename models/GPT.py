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
        if config.rank is None:
            # Multi Head Attention
            self.qkv_proj = nn.Linear(config.d_embed, 3 * config.d_embed, bias=config.attn_bias)
        else:
            # Multi Head Latent Attention
            assert config.rank < config.d_embed, "Rank must be less than embedding dimension"
            self.Wq = nn.Linear(config.d_embed, config.d_embed, bias=config.attn_bias)
            self.Wkv_down = nn.Linear(config.d_embed, config.rank, bias=False)
            self.Wkv_up = nn.Linear(config.rank, 2 * config.d_embed, bias=False)
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

        ########## Multi Head Attention ################################################################################
        if self.config.rank is None:
            # ---------- Linear projection -----------------------------------------------------------------------------
            q, k, v = self.qkv_proj(x).split(self.config.d_embed, dim=2)                # [batch_size, seq_len, d_embed]

            if kv_cache is not None:
                k_cache, v_cache = kv_cache                                         # kv_cache[0] -> k, kv_cache[1] -> v
                k = torch.cat([k_cache, k], dim=1)                               # [batch_size, seq_len, d_embed]
                v = torch.cat([v_cache, v], dim=1)                               # [batch_size, seq_len, d_embed]
            new_kv_cache = (k, v) if not self.training else None  # Only store cache if generation
            kv_seq_len = k.size(1)

        ########## Multi Head Latent Attention #########################################################################
        else:
            # ---------- Linear projection -------------------------------------------------------------------------
            q = self.Wq(x)                                                              # [batch_size, seq_len, d_embed]
            kv_latent = self.Wkv_down(x)                                                   # [batch_size, seq_len, rank]

            # KV cache
            if kv_cache is not None:
                kv_latent = torch.cat([kv_cache, kv_latent], dim=1)                 # [batch_size, seq_len, rank]
            new_kv_cache = kv_latent if not self.training else None  # Only store cache if generation
            kv_seq_len = kv_latent.size(1)

            k, v = self.Wkv_up(kv_latent).split(self.config.d_embed, dim=2)             # [batch_size, seq_len, d_embed]
        ################################################################################################################

        q = q.view(batch_size, seq_len, self.config.n_heads, self.config.d_head).transpose(1, 2)
        k = k.view(batch_size, kv_seq_len, self.config.n_heads, self.config.d_head).transpose(1, 2)
        v = v.view(batch_size, kv_seq_len, self.config.n_heads, self.config.d_head).transpose(1, 2)
                                                                                # [batch_size, n_heads, seq_len, d_head]

        # ---------- Casual self-attention -----------------------------------------------------------------------------
        if self.flash:
            attn = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal=True if seq_len > 1 else False
            )                                                                   # [batch_size, n_heads, seq_len, d_head]
        else:
            attn_scores = (q @ k.transpose(-2, -1)) * self.scale               # [batch_size, n_heads, seq_len, seq_len]
            attn_scores = attn_scores.masked_fill(self.mask[:, :, :seq_len, :kv_seq_len] == 0, float('-inf'))
            attn_scores = F.softmax(attn_scores, dim=-1)
            attn_scores = self.attn_dropout(attn_scores)
            attn = attn_scores @ v                                              # [batch_size, n_heads, seq_len, d_head]

        # ---------- Concatenation -------------------------------------------------------------------------------------
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.d_embed)
                                                                                        # [batch_size, seq_len, d_embed]

        # ---------- Output projection ---------------------------------------------------------------------------------
        attn_output = self.out_proj(attn)                                               # [batch_size, seq_len, d_embed]
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
        x = self.fc1(x)                                                                    # [batch_size, seq_len, d_ff]
        x = self.activation(x)
        x = self.fc2(x)                                                                 # [batch_size, seq_len, d_embed]
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
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(config.n_experts)])
        if config.n_shared_experts is not None:
            self.shared_experts = nn.ModuleList([FeedForward(config) for _ in range(config.n_shared_experts)])
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
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(config.n_experts)])

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
        self.inference_mode = False
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

    def forward(self, idx, targets=None, kv_cache=None):
        # Prefill
        if kv_cache is None:
            kv_cache = [None] * self.config.n_layers
            start_idx = 0
        # Decoding
        else:
            idx = idx[:, -1:]                                                                          # [batch_size, 1]
            # Multi Head Attention
            if self.config.rank is None:
                start_idx = kv_cache[0][0].size(1)
                # kv_cache[n] -> layer
                # kv_cache[n][0] -> k, kv_cache[n][1] -> v
                # k, v: [batch_size, seq_len, d_embed]
                if kv_cache[0][0].size(1) > self.config.max_seq_len - 1:
                    # RESET KV CACHE
                    print("Resetting KV cache")
                    kv_cache = [None] * self.config.n_layers
                    start_idx = 0
            # Multi Head Latent Attention
            else:
                start_idx = kv_cache[0].size(1)
                # kv_cache[n] -> layer
                # kv_latent: [batch_size, seq_len, rank]
                if kv_cache[0].size(1) > self.config.max_seq_len - 1:
                    # RESET KV CACHE
                    print("Resetting KV cache")
                    kv_cache = [None] * self.config.n_layers
                    start_idx = 0

        _, seq_len = idx.size()
        device = idx.device

        # ---------- Embedding -----------------------------------------------------------------------------------------
        tok_embed = self.token_embedding(idx)                                           # [batch_size, seq_len, d_embed]
        pos_idx = torch.arange(start_idx, start_idx + seq_len, device=device)                                # [seq_len]
        pos_embed = self.positional_encoding(pos_idx).unsqueeze(0)                               # [1, seq_len, d_embed]
        x = tok_embed + pos_embed                                                       # [batch_size, seq_len, d_embed]
        x = self.dropout(x)

        # ---------- Blocks --------------------------------------------------------------------------------------------
        new_kv_cache = []
        for layer_idx, block in enumerate(self.blocks):
            x, kv_cache_layer = block(x, kv_cache=kv_cache[layer_idx])                  # [batch_size, seq_len, d_embed]
            new_kv_cache.append(kv_cache_layer)

        # ---------- Final linear layer --------------------------------------------------------------------------------
        x = self.norm(x)
        # Training
        if targets is not None:
            logits = self.lm_head(x).view(-1, self.config.vocab_size)               # [batch_size * seq_len, vocab_size]
            targets = targets.view(-1)                                                          # [batch_size * seq_len]
            loss = F.cross_entropy(logits, targets, ignore_index=-1)
        # Generation
        else:
            logits = self.lm_head(x[:, -1:, :])                                            # [batch_size, 1, vocab_size]
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
        self.inference_mode = True
        if not (temperature > 0):
            raise ValueError("temperature must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        kv_cache = None
        for step in range(max_new_tokens):
            # ---------- Truncate --------------------------------------------------------------------------------------
            idx_input = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]

            # ---------- Forward pass ----------------------------------------------------------------------------------
            logits, _, kv_cache = self(
                idx_input,
                kv_cache=kv_cache if use_cache else None
            )                                                                              # [batch_size, 1, vocab_size]
            logits = logits[:, -1, :]                                                         # [batch_size, vocab_size]

            # ---------- Temperature -----------------------------------------------------------------------------------
            if temperature != 1.0:
                logits = logits / temperature

            # ---------- Top-k -----------------------------------------------------------------------------------------
            if top_k is not None:
                k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < k_logits[:, [-1]]] = -float('Inf')

            # ---------- Sample ----------------------------------------------------------------------------------------
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)                                         # [batch_size, 1]

            # ---------- Concatenate -----------------------------------------------------------------------------------
            idx = torch.cat((idx, next_idx), dim=1)                                  # [batch_size, seq_len + 1]

            # ---------- Streaming -------------------------------------------------------------------------------------
            if tokenizer is not None and idx.size(0) == 1:
                try:
                    next_str = tokenizer.decode(next_idx[0].tolist())
                    print(next_str, end='', flush=True)
                except Exception as e:
                    print(f"\nError decoding token: {e}")

        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) - self.lm_head.weight.numel()

    def num_active_params(self):
        pass

    def get_memory(self, x):
        num_params = self.num_params()
        # return the memory of the model
        # bytes per param = 2
        # also return the memory of data
        # bytes per

    def get_latency(self):
        pass

    def get_throughput(self):
        pass

    def get_flops(self, x):
        B, T = x.size()
        D = self.config.d_embed

        # ---------- Accelerator Intensity in Nvidia --------------------------------------------------------------------------
        # FLOPs = 52.22 TFLOPs
        # Memory Bandwidth = 736.6GB/s
        # TensorCore Accelerator Intensity = 70.65

        # ---------- Matrix multiplication FLOPs and Memory read/writes ------------------------------------------------
        # x = [B, T, D], W = [D, F]
        # 1. Read x to SRAM
        # bytes = 2 x B X T X D = 2BTD
        # 2. Read W to SRAM
        # bytes = 2 x D x F = 2DF
        # 3. Compute Y = x @ W
        # FLOPS = 2 x B x T x D x F = 2BTDF
        # 4. Write Y to HBM
        # bytes = 2BTF
        # Arithmetic Intensity
        # AI = 2BTDF / (2BTD + 2DF + 2BTF) = 4BTD / (5BT + 4D)
        #    = BT

        # ---------- Attention FLOPs and Memory read/writes ------------------------------------------------------------
        # 1. Read x from HBM
        # bytes =
        # 2. Read Wq Wk Wv from HBM
        # bytes =
        # 3. Compute
        #
        #

        ## Prefill
        ######### Flash Attention #########
        # 1. Read Q, K, V from HBM
        # bytes = 3 x (2 x B x T x D) = 6BTD
        # 2. Compute Q @ K
        # FLOPS = 2 x B x T x T x D = 2BT^2D
        # 3. Compute A @ V
        # FLOPS = 2 × B × T × T × D = 2BT^2D
        # 4. Write attn_out
        # bytes = 2 x B x T x D = 2BTD
        ####################################
        # attn bytes = 6BTD + 2BTD = 8BTD
        # attn FLOPS = 2BT^2D + 2BT^2D = 4BT^2D
        # attn AI = 4BT^2D / 8BTD = T/2
        #         = T
        attn_prefill_flops = 4 * B * T * T * D
        attn_prefill_ai = T / 2

        ## Decoding
        ######### Flash Attention #########
        # 1. Read Q, K, V from HBM
        # bytes = 2 x B x 1 x D + 2 x (2 x B x S x D) = 2BD + 4BSD
        # 2. Compute Q @ K
        # FLOPS = 2 x B x 1 x S x D = 2BSD
        # 3. Compute A @ V
        # FLOPS = 2 × B × S × 1 × D = 2BSD
        # 4. Write attn_out
        # bytes = 2 x B x 1 x D = 2BD
        ####################################
        # attn bytes = 2BD + 4BSD + 2BD = 4BD(1 + S)
        # attn FLOPS = 2BSD + 2BSD = 4BSD
        # attn AI = 4BSD / 4BD(1 + S) = S / (1 + S) (ignore 1)
        #         = 1
        attn_decoding_flops = 4 * B * T * D
        attn_decoding_ai = T / (1 + T)
        # Why so low ai?
        # FLOPS decreased by T, bytes remains the same

        ## KV cache
        ##### MultiHeadAttention ###########
        # size = 2 x (2 x B x S x D) = 4BSD
        # AI = 1
        ##### GroupedQueryAttention ########
        # size = 2 x (2 x B x S x D / n_groups) = 4BSD / n_groups
        # AI = G (group_size)
        ##### MultiQueryAttention ##########
        # size = 2 x (2 x B X S x d) = 4BSD / n_heads
        # AI = n_heads
        ##### MultiHeadLatentAttention #####
        # size = 2 x (B x S x R) = 2BSR = 4BSD / (2D/R)
        # AI = 2D/R

        # ---------- FeedForward FLOPs and Memory read/writes ----------------------------------------------------------
        # 1. Read x from HBM
        # bytes = 2 x B x T x D = 2BTD
        # 2. Read Wup, Wdown from HBM
        # bytes = 2 x (2 x D x 4D) = 16D^2
        # 3. Compute x @ Wup
        # FLOPS = 2 x B x T x D x 4D = 8BTD^2
        # 4. Compute x @ Wdown
        # FLOPS = 2 x B x T x 4D x D = 8BTD^2
        # 5. Write x to HBM
        # bytes = 2 x B x T x D = 2BTD
        ####################################
        # FF bytes = 2BTD + 16D^2 + 2BTD = 4D(BT + 4D)
        # FF FLOPS = 8BTD^2 + 8BTD^2 = 16BTD^2
        # FF AI = 16BTD^2 / 4D(BT + 4D) = 4BTD / (BT + 4D) (ignore 2BT)
        #       = BT
        feedforward_flops = 16 * B * T * D * D
        feedforward_ai = 4 * B * T * D / (B * T + 4 * D)

        flops = {'attn_prefill_flops': attn_prefill_flops, 'attn_prefill_ai': attn_prefill_ai,
                 'attn_decoding_flops': attn_decoding_flops, 'attn_decoding_ai': attn_decoding_ai,
                 'ff_flops': feedforward_flops, 'ff_ai': feedforward_ai}

        return flops


def main():
    model_config = ModelConfig()
    model = GPT(model_config)
    print(model)
    print(f"Number of parameters: {model.num_params() / 1e6:.2f}M")


if __name__ == "__main__":
    main()