import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from src.config import ModelConfig


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        assert config.d_embed % config.n_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.config = config
        self.layer_idx = layer_idx
        if config.rank is None:
            # Multi Head Attention
            self.qkv_proj = nn.Linear(config.d_embed, 3 * config.d_embed, bias=config.attn_bias)
        else:
            # Multi Head Latent Attention
            assert config.rank < config.d_embed, "Rank must be less than embedding dimension"
            self.Wq = nn.Linear(config.d_embed, config.d_embed, bias=False)
            self.Wkv_down = nn.Linear(config.d_embed, config.rank, bias=False)
            self.Wk_up = nn.Linear(config.rank, config.d_embed, bias=False)
            self.Wv_up = nn.Linear(config.rank, config.d_embed, bias=False)
        self.out_proj = nn.Linear(config.d_embed, config.d_embed, bias=config.attn_bias)
        self.dropout = nn.Dropout(config.dropout)

        if config.flash is True:
            assert hasattr(F, "scaled_dot_product_attention"), "Flash attention requires PyTorch 2.0 or higher"
        else:
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

            # TODO
            # ---------- Rotary positional embeddings ------------------------------------------------------------------


            # ---------- KV cache  -------------------------------------------------------------------------------------
            if kv_cache is not None:
                k_cache, v_cache = kv_cache                                         # kv_cache[0] -> k, kv_cache[1] -> v
                k = torch.cat([k_cache, k], dim=1)                               # [batch_size, seq_len, d_embed]
                v = torch.cat([v_cache, v], dim=1)                               # [batch_size, seq_len, d_embed]
            new_kv_cache = (k, v) if not self.training else None # Only store cache if generation
            kv_seq_len = k.size(1)

            q = q.view(batch_size, seq_len, self.config.n_heads, self.config.d_head).transpose(1, 2)
            k = k.view(batch_size, kv_seq_len, self.config.n_heads, self.config.d_head).transpose(1, 2)
            v = v.view(batch_size, kv_seq_len, self.config.n_heads, self.config.d_head).transpose(1, 2)
                                                                                # [batch_size, n_heads, seq_len, d_head]

            # ---------- Causal self-attention -------------------------------------------------------------------------
            if self.config.flash:
                attn_out = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.config.dropout if self.training else 0.0,
                    is_causal=True if seq_len > 1 else False
                )                                                               # [batch_size, n_heads, seq_len, d_head]
            else:
                attn_scores = (q @ k.transpose(-2, -1)) * self.scale           # [batch_size, n_heads, seq_len, seq_len]
                attn_scores = attn_scores.masked_fill(self.mask[:, :, :seq_len, :kv_seq_len] == 0, float('-inf'))
                attn_scores = F.softmax(attn_scores, dim=-1)
                attn_scores = self.attn_dropout(attn_scores)
                attn_out = attn_scores @ v                                      # [batch_size, n_heads, seq_len, d_head]

            # ---------- Concatenation ---------------------------------------------------------------------------------
            attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.d_embed)
                                                                                        # [batch_size, seq_len, d_embed]

            # ---------- Output projection -----------------------------------------------------------------------------
            y = self.out_proj(attn_out)                                                 # [batch_size, seq_len, d_embed]
            y = self.dropout(y)
            return y, new_kv_cache

        ########## Multi Head Latent Attention #########################################################################
        else:
            if not self.config.cross_layer_attention:
                if self.training:
                    # ---------- Linear projection ---------------------------------------------------------------------
                    q = self.Wq(x)                                                      # [batch_size, seq_len, d_embed]
                    kv_latent = self.Wkv_down(x)                                           # [batch_size, seq_len, rank]
                    k = self.Wk_up(kv_latent)                                           # [batch_size, seq_len, d_embed]
                    v = self.Wv_up(kv_latent)                                           # [batch_size, seq_len, d_embed]

                    q = q.view(batch_size, seq_len, self.config.n_heads, self.config.d_head).transpose(1, 2)
                    k = k.view(batch_size, seq_len, self.config.n_heads, self.config.d_head).transpose(1, 2)
                    v = v.view(batch_size, seq_len, self.config.n_heads, self.config.d_head).transpose(1, 2)
                                                                                # [batch_size, n_heads, seq_len, d_head]

                    # ---------- Causal self-attention ---------------------------------------------------------------------
                    if self.config.flash:
                        attn_out = F.scaled_dot_product_attention(
                            q, k, v,
                            dropout_p=self.config.dropout,
                            is_causal=True
                        )                                                       # [batch_size, n_heads, seq_len, d_head]
                    else:
                        attn_scores = (q @ k.transpose(-2, -1)) * self.scale   # [batch_size, n_heads, seq_len, seq_len]
                        attn_scores = attn_scores.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))
                        attn_scores = F.softmax(attn_scores, dim=-1)
                        attn_scores = self.attn_dropout(attn_scores)
                        attn_out = attn_scores @ v                              # [batch_size, n_heads, seq_len, d_head]

                    # ---------- Concatenation -----------------------------------------------------------------------------
                    attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.d_embed)
                                                                                        # [batch_size, seq_len, d_embed]

                    # ---------- Output projection ---------------------------------------------------------------------
                    y = self.out_proj(attn_out)                                         # [batch_size, seq_len, d_embed]
                    y = self.dropout(y)
                    return y, None

                else:
                    # ---------- Linear projection ---------------------------------------------------------------------
                    q = self.Wq(x)                                                      # [batch_size, seq_len, d_embed]
                    q = q.view(batch_size, seq_len, self.config.n_heads, self.config.d_head).transpose(1, 2)
                                                                                # [batch_size, n_heads, seq_len, d_head]
                    Wk_up = self.Wk_up.weight.view(self.config.n_heads, self.config.d_head, self.config.rank)
                    q_latent = q @ Wk_up                                          # [batch_size, n_heads, seq_len, rank]
                    kv_latent = self.Wkv_down(x)                                           # [batch_size, seq_len, rank]

                    # ---------- KV cache  -----------------------------------------------------------------------------
                    if kv_cache is not None:
                        kv_latent = torch.cat([kv_cache, kv_latent], dim=1)        # [batch_size, seq_len, rank]
                    new_kv_cache = kv_latent
                    kv_seq_len = kv_latent.size(1)

                    # ---------- Causal self-attention -----------------------------------------------------------------
                    kv_latent = kv_latent.unsqueeze(1)                                  # [batch_size, 1, seq_len, rank]
                    kv_latent = kv_latent.repeat(1, self.config.n_heads, 1, 1)    # [batch_size, n_heads, seq_len, rank]
                    if self.config.flash:
                        attn = F.scaled_dot_product_attention(
                            q_latent, kv_latent, kv_latent,
                            is_causal=True if seq_len > 1 else False
                        )                                                         # [batch_size, n_heads, seq_len, rank]
                    else:
                        attn_scores = (q_latent @ kv_latent.transpose(-2, -1)) * self.scale
                        attn_scores = attn_scores.masked_fill(self.mask[:, :, :seq_len, :kv_seq_len] == 0, float('-inf'))
                        attn_scores = F.softmax(attn_scores, dim=-1)
                        attn = attn_scores @ kv_latent                            # [batch_size, n_heads, seq_len, rank]
                    Wv_up = self.Wv_up.weight.view(self.config.n_heads, self.config.d_head, self.config.rank).transpose(1, 2)
                    attn = attn @ Wv_up                                         # [batch_size, n_heads, seq_len, d_head]

                    # ---------- Concatenation -------------------------------------------------------------------------
                    attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.d_embed)
                                                                                        # [batch_size, seq_len, d_embed]

                    # ---------- Output projection ---------------------------------------------------------------------
                    attn_output = self.out_proj(attn)                                   # [batch_size, seq_len, d_embed]
                    return attn_output, new_kv_cache
            # Cross-Layer Latent Attention
            else:
                if self.training:
                    # ---------- Linear projection ---------------------------------------------------------------------
                    q = self.Wq(x)                                                      # [batch_size, seq_len, d_embed]
                    if self.layer_idx == 0:
                        kv_latent = self.Wkv_down(x)                                       # [batch_size, seq_len, rank]
                    else: # Layer 1+
                        kv_latent = kv_cache                                               # [batch_size, seq_len, rank]
                    k = self.Wk_up(kv_latent)                                           # [batch_size, seq_len, d_embed]
                    v = self.Wv_up(kv_latent)                                           # [batch_size, seq_len, d_embed]

                    q = q.view(batch_size, seq_len, self.config.n_heads, self.config.d_head).transpose(1, 2)
                    k = k.view(batch_size, seq_len, self.config.n_heads, self.config.d_head).transpose(1, 2)
                    v = v.view(batch_size, seq_len, self.config.n_heads, self.config.d_head).transpose(1, 2)
                                                                                # [batch_size, n_heads, seq_len, d_head]

                    # ---------- Causal self-attention -----------------------------------------------------------------
                    if self.config.flash:
                        attn_out = F.scaled_dot_product_attention(
                            q, k, v,
                            dropout_p=self.config.dropout,
                            is_causal=True
                        )                                                       # [batch_size, n_heads, seq_len, d_head]
                    else:
                        attn_scores = (q @ k.transpose(-2, -1)) * self.scale   # [batch_size, n_heads, seq_len, seq_len]
                        attn_scores = attn_scores.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))
                        attn_scores = F.softmax(attn_scores, dim=-1)
                        attn_scores = self.attn_dropout(attn_scores)
                        attn_out = attn_scores @ v                              # [batch_size, n_heads, seq_len, d_head]

                    # ---------- Concatenation -------------------------------------------------------------------------
                    attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.d_embed)
                                                                                        # [batch_size, seq_len, d_embed]

                    # ---------- Output projection ---------------------------------------------------------------------
                    y = self.out_proj(attn_out)                                         # [batch_size, seq_len, d_embed]
                    y = self.dropout(y)
                    return y, kv_latent

                else:
                    # ---------- Linear projection ---------------------------------------------------------------------
                    q = self.Wq(x)                                                      # [batch_size, seq_len, d_embed]
                    q = q.view(batch_size, seq_len, self.config.n_heads, self.config.d_head).transpose(1, 2)
                                                                                # [batch_size, n_heads, seq_len, d_head]
                    Wk_up = self.Wk_up.weight.view(self.config.n_heads, self.config.d_head, self.config.rank)
                    q_latent = q @ Wk_up                                          # [batch_size, n_heads, seq_len, rank]

                    # ---------- KV cache  -----------------------------------------------------------------------------
                    if self.layer_idx == 0:
                        kv_latent = self.Wkv_down(x)                                       # [batch_size, seq_len, rank]
                        if kv_cache is not None:
                            kv_latent = torch.cat([kv_cache, kv_latent], dim=1)    # [batch_size, seq_len, rank]
                    else: # Layer 1+
                        kv_latent = kv_cache                                               # [batch_size, seq_len, rank]
                    new_kv_cache = kv_latent
                    kv_seq_len = kv_latent.size(1)

                    # ---------- Causal self-attention -----------------------------------------------------------------
                    kv_latent = kv_latent.unsqueeze(1)                                  # [batch_size, 1, seq_len, rank]
                    kv_latent = kv_latent.repeat(1, self.config.n_heads, 1, 1)    # [batch_size, n_heads, seq_len, rank]
                    if self.config.flash:
                        attn = F.scaled_dot_product_attention(
                            q_latent, kv_latent, kv_latent,
                            is_causal=True if seq_len > 1 else False
                        )                                                         # [batch_size, n_heads, seq_len, rank]
                    else:
                        attn_scores = (q_latent @ kv_latent.transpose(-2, -1)) * self.scale
                        attn_scores = attn_scores.masked_fill(self.mask[:, :, :seq_len, :kv_seq_len] == 0, float('-inf'))
                        attn_scores = F.softmax(attn_scores, dim=-1)
                        attn = attn_scores @ kv_latent                            # [batch_size, n_heads, seq_len, rank]
                    Wv_up = self.Wv_up.weight.view(self.config.n_heads, self.config.d_head, self.config.rank).transpose(1, 2)
                    attn = attn @ Wv_up                                         # [batch_size, n_heads, seq_len, d_head]

                    # ---------- Concatenation -------------------------------------------------------------------------
                    attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.d_embed)
                                                                                        # [batch_size, seq_len, d_embed]

                    # ---------- Output projection ---------------------------------------------------------------------
                    attn_output = self.out_proj(attn)                                   # [batch_size, seq_len, d_embed]
                    return attn_output, new_kv_cache


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        if config.beta_min is None and config.beta_max is None:
            d_ff = config.d_ff
            d_ff = config.d_ff_multiple_of * ((d_ff + config.d_ff_multiple_of - 1) // config.d_ff_multiple_of)
        # Layer-wise scaling
        else:
            beta = config.beta_min + (config.beta_max - config.beta_min) * layer_idx / (config.n_layers - 1)
            d_ff = int(config.d_embed * beta)
            d_ff = config.d_ff_multiple_of * ((d_ff + config.d_ff_multiple_of - 1) // config.d_ff_multiple_of)

        self.fc1 = nn.Linear(config.d_embed, d_ff, bias=config.mlp_bias)
        self.fc2 = nn.Linear(d_ff, config.d_embed, bias=config.mlp_bias)
        self.activation = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
        }[config.activation]()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)                                                                    # [batch_size, seq_len, d_ff]
        x = self.activation(x)
        x = self.fc2(x)                                                                 # [batch_size, seq_len, d_embed]
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_embed, eps=config.norm_eps)
        self.attn = MultiHeadAttention(config, layer_idx=layer_idx)
        self.norm2 = nn.LayerNorm(config.d_embed, eps=config.norm_eps)
        self.mlp = FeedForward(config, layer_idx=layer_idx)

    def forward(self, x, kv_cache):
        x = self.norm1(x)
        y, new_kv_cache = self.attn(x, kv_cache=kv_cache)
        x = x + y

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
        self.blocks = nn.ModuleList([
            Block(config, layer_idx=i)
            for i in range(config.n_layers)
        ])
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

    def forward(self, input_ids, target_ids=None, kv_cache=None):
        # Prefill
        if kv_cache is None:
            if not self.config.cross_layer_attention:
                kv_cache = [None] * self.config.n_layers
            else:
                kv_cache = None
            start_idx = 0

        # Decoding
        else:
            input_ids = input_ids[:, -1:]                                                              # [batch_size, 1]

            if self.config.rank is None:                                                          # Multi Head Attention
                start_idx = kv_cache[0][0].size(1)
                # kv_cache[n] -> layer
                # kv_cache[n][0] -> k, kv_cache[n][1] -> v
                # k, v: [batch_size, seq_len, d_embed]
                if kv_cache[0][0].size(1) > self.config.max_seq_len - 1:
                    # RESET KV CACHE
                    print("Resetting KV cache")
                    kv_cache = [None] * self.config.n_layers
                    start_idx = 0

            else:                                                                          # Multi Head Latent Attention
                if not self.config.cross_layer_attention:
                    start_idx = kv_cache[0].size(1)
                    # kv_cache[n] -> layer
                    # kv_cache[n]: [batch_size, seq_len, rank]
                    if kv_cache[0].size(1) > self.config.max_seq_len - 1:
                        # RESET KV CACHE
                        print("Resetting KV cache")
                        kv_cache = [None] * self.config.n_layers
                        start_idx = 0
                else:                                                          # Multi-Head Cross-Layer Latent Attention
                    start_idx = kv_cache.size(1)
                    # kv_cache: [batch_size, seq_len, rank]
                    if kv_cache.size(1) > self.config.max_seq_len - 1:
                        # RESET KV CACHE
                        print("Resetting KV cache")
                        kv_cache = None
                        start_idx = 0

        _, seq_len = input_ids.size()
        device = input_ids.device

        # ---------- Embedding -----------------------------------------------------------------------------------------
        tok_embed = self.token_embedding(input_ids)                                     # [batch_size, seq_len, d_embed]
        pos_idx = torch.arange(start_idx, start_idx + seq_len, device=device)                                # [seq_len]
        pos_embed = self.positional_encoding(pos_idx).unsqueeze(0)                               # [1, seq_len, d_embed]
        x = tok_embed + pos_embed                                                       # [batch_size, seq_len, d_embed]
        x = self.dropout(x)

        # ---------- Blocks --------------------------------------------------------------------------------------------
        new_kv_cache = []
        if not self.config.cross_layer_attention:
            for layer_idx, block in enumerate(self.blocks):
                x, kv_cache_layer = block(x, kv_cache=kv_cache[layer_idx])              # [batch_size, seq_len, d_embed]
                new_kv_cache.append(kv_cache_layer)
        else:
            for layer_idx, block in enumerate(self.blocks):
                x, kv_cache = block(x, kv_cache=kv_cache)                               # [batch_size, seq_len, d_embed]
            new_kv_cache = kv_cache

        # ---------- Final linear layer --------------------------------------------------------------------------------
        x = self.norm(x)
        # Training
        if target_ids is not None:
            logits = self.lm_head(x).view(-1, self.config.vocab_size)               # [batch_size * seq_len, vocab_size]
            targets = target_ids.view(-1)                                                       # [batch_size * seq_len]
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
            idx = torch.cat((idx, next_idx), dim=1)                                   # [batch_size, seq_len + 1]

            # ---------- Streaming -------------------------------------------------------------------------------------
            if tokenizer is not None and idx.size(0) == 1:
                try:
                    next_str = tokenizer.decode(next_idx[0].tolist())
                    print(next_str, end='', flush=True)
                except Exception as e:
                    print(f"\nError decoding token: {e}")

        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # TODO
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

        # ---------- Accelerator Intensity in Nvidia -------------------------------------------------------------------
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

        ########## Multi Head Attention ################################################################################
        if self.config.rank is None:
            ## Prefill
            ######### Flash Attention ##########
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
            ######### Flash Attention ##########
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
            # FLOPS decreased by T, bytes remain the same

        ########## Multi Head Latent Attention #########################################################################
        else:
            R = self.config.rank
            ## Prefill
            # 1. Read KV_latent from HBM
            # bytes = 2 x (2 x B x T x R) = 4BTR
            # 2. Read Wkv_up from HBM
            # bytes = 2 x R x 2D = 4RD
            # 3. Compute kv_latent @ Wkv_up
            # FLOPS = 2 x B x T x R x 2D = 4BTRD
            # 4. Write k, v to HBM
            # bytes = 2 x (2 x B x T x D) = 4BTD
            ######### Flash Attention ##########
            # 1. Read Q, K, V from HBM
            # bytes = 3 x (2 x B x T x D) = 6BTD
            # 2. Compute Q @ K
            # FLOPS = 2 x B x T x T x D = 2BT^2D
            # 3. Compute A @ V
            # FLOPS = 2 × B × T × T × D = 2BT^2D
            # 4. Write y
            # bytes = 2 x B x T x D = 2BTD
            ####################################
            # attn bytes = 4BTR + 4RD + 4BTD + 6BTD + 2BTD = 4(RD + BTR + 3BTD)
            # attn FLOPS = 4BTRD + 2BT^2D + 2BT^2D = 4BTD(R + T)
            # attn AI = 4BTD(R + T) / 4(RD + BTR + 3BTD) = BTD(R + T) / RD + BTR + 3BTD
            #         =
            attn_prefill_flops = 4 * B * T * D * (R + T)
            attn_prefill_ai = T / 2

            ## Decoding
            # 1. KV_latent from HBM
            # bytes = 2 x (2 x B x S x R) = 4BSR
            # 2. Read Wkv_up from HBM
            # bytes = 2 x R x 2D = 4RD
            # 3. Compute kv_latent @ Wkv_up
            # FLOPS = 2 x B x S x R x 2D = 4BSRD
            # 4. Write k, v to HBM
            # bytes = 2 x (2 x B x S x D) = 4BSD
            ######### Flash Attention ##########
            # 1. Read Q_latent, KV_latent from HBM
            # bytes = 2 x B x S x R x D / d + 2 x B x S x R = 2BSR(D/d + 1)
            # 2. Compute Q_latent @ K_latent
            # FLOPS = 2 x B x 1 x S x R = 2BSR
            # 3. Compute A @ KV_latent
            # FLOPS = 2 x B x S x 1 x R = 2BSR
            # 4. Write y
            # bytes = 2 x B x 1 X R = 2BR
            ####################################
            # 5. Compute
            # attn bytes =
            # attn FLOPS =
            # attn AI = 2S / (1 + S(D/d + 1))
            #         =
            attn_decoding_flops = 4 * B * T * D
            attn_decoding_ai = T / (1 + T)
        ################################################################################################################

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

        flops = {
            'attn_prefill_flops': attn_prefill_flops, 'attn_prefill_ai': attn_prefill_ai,
            'attn_decoding_flops': attn_decoding_flops, 'attn_decoding_ai': attn_decoding_ai,
            'feedforward_flops': feedforward_flops, 'feedforward_ai': feedforward_ai
        }

        return flops


def main():
    model_config = ModelConfig(
        d_embed=1024,
        n_layers=10,
        n_heads=16,
        d_head=64,
        rank=32,
        d_ff=-1,
        beta_min=1 / 2,
        beta_max=4,
        cross_layer_attention=True
    )
    model = GPT(model_config)
    print(model)
    print(f"Number of parameters: {model.num_params() / 1e6:.2f}M")

    #sample_input = torch.randint(0, model_config.vocab_size, (1, 1))
    #print(f"Attention Prefill


if __name__ == "__main__":
    main()