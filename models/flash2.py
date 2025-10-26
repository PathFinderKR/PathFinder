import time
import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.runtime import driver
from src.utils import set_seed
BLOCK_T = 128
BLOCK_D = 128
NUM_KV_SPLIT = 8
NUM_WARPS = 8
NUM_STAGES = 3


def naive_attention(q, k, v):
    scale = 1.0 / math.sqrt(q.size(-1))
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_prob = torch.softmax(attn_scores, dim=-1)
    attn_out = torch.matmul(attn_prob, v)
    return attn_out

@triton.jit
def flash_attn_2_kernel(
    Q,     # [B, H, 1, D] (bf16)
    K, V,  # [B, H, T, D] (bf16)
    O,     # [B, H, 1, D] (bf16)
    stride_q_b, stride_q_h, stride_q_t, stride_q_d,
    stride_k_b, stride_k_h, stride_k_t, stride_k_d,
    stride_v_b, stride_v_h, stride_v_t, stride_v_d,
    stride_o_b, stride_o_h, stride_o_t, stride_o_d,
    B, H, T, D,
    N_BLOCKS: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Program IDs
    off_hb = tl.program_id(0)
    off_h  = off_hb % H
    off_b  = off_hb // H

    # Offsets
    offs_t = tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)

    # Pointers
    q_ptrs = Q + off_b * stride_q_b + off_h * stride_q_h + 0 * stride_q_t + offs_d * stride_q_d
    k_base = K + off_b * stride_k_b + off_h * stride_k_h
    v_base = V + off_b * stride_v_b + off_h * stride_v_h
    o_ptrs = O + off_b * stride_o_b + off_h * stride_o_h + 0 * stride_o_t + offs_d * stride_o_d

    # Load query vector
    q = tl.load(q_ptrs, mask=offs_d < D, other=0.0).to(tl.float32)  # [BLOCK_D]

    # Constants
    Df      = tl.full((), 0.0, dtype=tl.float32) + D
    scale   = 1.0 / tl.sqrt(Df)
    neg_inf = tl.full((), -float("inf"), dtype=tl.float32)

    # online softmax stats (fp32)
    max_score = neg_inf
    sum_exp   = tl.zeros((), dtype=tl.float32)
    acc       = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Loop over K, V blocks along T
    for b in range(0, N_BLOCKS):
        start_n = b * BLOCK_T
        t_idx = start_n + offs_t  # [BLOCK_T]
        mask_t = t_idx < T        # [BLOCK_T]
        mask_d = offs_d < D       # [BLOCK_D]

        # Pointers
        k_ptrs = k_base + t_idx[:, None] * stride_k_t + offs_d[None, :] * stride_k_d
        v_ptrs = v_base + t_idx[:, None] * stride_v_t + offs_d[None, :] * stride_v_d

        # Load K, V blocks (bf16)
        kv_mask = mask_t[:, None] & mask_d[None, :]
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)  # [BLOCK_T, BLOCK_D]
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)  # [BLOCK_T, BLOCK_D]

        # Compute attention scores: q @ K^T
        scores = tl.sum(k * q[None, :], axis=1) * scale  # [BLOCK_T] (fp32)
        scores = tl.where(mask_t, scores, neg_inf)

        # Online-softmax update (fp32)
        block_max = tl.max(scores, axis=0)
        new_max   = tl.maximum(max_score, block_max)
        rescale   = tl.where(max_score == neg_inf, 0.0, tl.exp(max_score - new_max))
        acc       *= rescale
        sum_exp   *= rescale

        # Probabilities
        safe_scores = tl.where(mask_t, scores, new_max)
        probs = tl.exp(safe_scores - new_max)
        probs = tl.where(mask_t, probs, 0.0)

        # Update accumulator and sum
        acc       += tl.sum(probs[:, None] * v, axis=0)  # [BLOCK_D]
        sum_exp   += tl.sum(probs, axis=0)
        max_score = new_max

    # Final normalization
    out = acc / tl.maximum(sum_exp, 1e-20)

    # Store output
    tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=offs_d < D)

def flash_attn_2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Flash Attention 2 Triton Kernel

    Args:
        q: Query tensor [batch, n_heads, 1, d_head]
        k: Key tensor   [batch, n_heads, seq_len_k, d_head]
        v: Value tensor [batch, n_heads, seq_len_v, d_head]

    Returns:
        o: Output tensor [batch, n_heads, 1, d_head]
    """
    _, _, q_len, _ = q.shape
    batch_size, n_heads, kv_seq_len, d_heads = k.shape
    assert q.ndim == k.ndim == v.ndim == 4
    assert q_len == 1

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    o = torch.empty_like(q)

    grid = (batch_size * n_heads, )
    num_blocks = triton.cdiv(kv_seq_len, BLOCK_T)
    flash_attn_2_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        batch_size, n_heads, kv_seq_len, d_heads,
        N_BLOCKS=num_blocks,
        BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES
    )
    return o


@triton.jit
def flash_attn_2_fixmax_kernel(
    Q,      # [B, H, 1, D] (bf16)
    K, V,   # [B, H, T, D] (bf16)
    O,      # [B, H, 1, D] (bf16)
    stride_q_b, stride_q_h, stride_q_t, stride_q_d,
    stride_k_b, stride_k_h, stride_k_t, stride_k_d,
    stride_v_b, stride_v_h, stride_v_t, stride_v_d,
    stride_o_b, stride_o_h, stride_o_t, stride_o_d,
    B, H, T, D,
    FIXMAX: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Program IDs
    off_hb = tl.program_id(0)
    off_h  = off_hb % H
    off_b  = off_hb // H

    # Offsets
    offs_t = tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)

    # Pointers
    q_ptrs = Q + off_b * stride_q_b + off_h * stride_q_h + 0 * stride_q_t + offs_d * stride_q_d
    k_base = K + off_b * stride_k_b + off_h * stride_k_h
    v_base = V + off_b * stride_v_b + off_h * stride_v_h
    o_ptrs = O + off_b * stride_o_b + off_h * stride_o_h + 0 * stride_o_t + offs_d * stride_o_d

    # Load query vector
    q = tl.load(q_ptrs, mask=offs_d < D, other=0.0).to(tl.float32)  # [BLOCK_D]

    # Constants
    Df      = tl.full((), 0.0, dtype=tl.float32) + D
    scale   = 1.0 / tl.sqrt(Df)
    neg_inf = tl.full((), -float("inf"), dtype=tl.float32)

    # online softmax stats (fp32)
    sum_exp = tl.zeros((), dtype=tl.float32)
    acc     = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Fixmax (fp32)
    fixmax = tl.full((), FIXMAX, dtype=tl.float32)

    # Loop over K, V blocks along T
    for b in range(0, N_BLOCKS):
        start_n = b * BLOCK_T
        t_idx = start_n + offs_t  # [BLOCK_T]
        mask_t = t_idx < T        # [BLOCK_T]
        mask_d = offs_d < D       # [BLOCK_D]

        # Pointers
        k_ptrs = k_base + t_idx[:, None] * stride_k_t + offs_d[None, :] * stride_k_d
        v_ptrs = v_base + t_idx[:, None] * stride_v_t + offs_d[None, :] * stride_v_d

        # Load K, V blocks (bf16)
        kv_mask = mask_t[:, None] & mask_d[None, :]
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)  # [BLOCK_T, BLOCK_D]
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)  # [BLOCK_T, BLOCK_D]

        # Compute attention scores: q @ K^T
        scores = tl.sum(k * q[None, :], axis=1) * scale  # [BLOCK_T] (fp32)
        scores = tl.where(mask_t, scores, neg_inf)

        # Fixed-max softmax probs
        probs = tl.exp(scores - fixmax)  # [BLOCK_T]
        probs = tl.where(mask_t, probs, 0.0)

        # Update accumulator and sum
        acc     += tl.sum(probs[:, None] * v, axis=0)  # [BLOCK_D]
        sum_exp += tl.sum(probs, axis=0)

    # Final normalization
    out = acc / tl.maximum(sum_exp, 1e-20)

    # Store output
    tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=offs_d < D)

def flash_attn_2_fixmax(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    fixmax: float = 10
) -> torch.Tensor:
    """
    Flash Attention 2 Triton Kernel

    Args:
        q: Query tensor [batch, n_heads, 1, d_head]
        k: Key tensor   [batch, n_heads, seq_len_k, d_head]
        v: Value tensor [batch, n_heads, seq_len_v, d_head]
        fixmax: scalar

    Returns:
        o: Output tensor [batch, n_heads, 1, d_head]
    """
    _, _, q_len, _ = q.shape
    batch_size, n_heads, kv_seq_len, d_heads = k.shape
    assert q.ndim == k.ndim == v.ndim == 4
    assert q_len == 1

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    o = torch.empty_like(q)

    grid = (batch_size * n_heads, )
    num_blocks = triton.cdiv(kv_seq_len, BLOCK_T)
    flash_attn_2_fixmax_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        batch_size, n_heads, kv_seq_len, d_heads,
        fixmax,
        N_BLOCKS=num_blocks,
        BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES
    )
    return o


@triton.jit
def flash_decoding_pass1(
    Q,     # [B, H, 1, D] (bf16)
    K, V,  # [B, H, T, D] (bf16)
    M,     # [B, H, P]    (fp32)  partition max
    L,     # [B, H, P]    (fp32)  partition sum-exp
    A,     # [B, H, P, D] (fp32)  partition accumulator over D
    stride_q_b, stride_q_h, stride_q_t, stride_q_d,
    stride_k_b, stride_k_h, stride_k_t, stride_k_d,
    stride_v_b, stride_v_h, stride_v_t, stride_v_d,
    stride_m_b, stride_m_h, stride_m_p,
    stride_l_b, stride_l_h, stride_l_p,
    stride_a_b, stride_a_h, stride_a_p, stride_a_d,
    B, H, T, D,
    T_PART: tl.constexpr,    # ceil_div(T, P)
    N_BLOCKS: tl.constexpr,  # ceil_div(T_PART, BLOCK_T)
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # Program IDs
    pid_hb = tl.program_id(0)
    pid_p  = tl.program_id(1)
    off_h  = pid_hb % H
    off_b  = pid_hb // H

    # Offsets
    offs_t = tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)

    # Pointers
    q_ptrs = Q + off_b * stride_q_b + off_h * stride_q_h + 0 * stride_q_t + offs_d * stride_q_d
    k_base = K + off_b * stride_k_b + off_h * stride_k_h
    v_base = V + off_b * stride_v_b + off_h * stride_v_h

    # Load query vector
    q = tl.load(q_ptrs, mask=offs_d < D, other=0.0).to(tl.float32)  # [BLOCK_D]

    # partition bounds
    t_start  = pid_p * T_PART
    t_end    = tl.minimum(t_start + T_PART, T)
    is_empty = t_start >= T

    # Constants
    Df      = tl.full((), 0.0, dtype=tl.float32) + D
    scale   = 1.0 / tl.sqrt(Df)
    neg_inf = tl.full((), -float("inf"), dtype=tl.float32)

    # online softmax stats (fp32)
    max_score = neg_inf
    sum_exp   = tl.zeros((), dtype=tl.float32)
    acc       = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Loop over K, V blocks along T-partition
    for b in range(0, N_BLOCKS):
        tile_t = t_start + b * BLOCK_T + offs_t
        mask_t = (tile_t < t_end) & (~is_empty)

        # Pointers
        k_ptrs = k_base + tile_t[:, None]*stride_k_t + offs_d[None, :]*stride_k_d
        v_ptrs = v_base + tile_t[:, None]*stride_v_t + offs_d[None, :]*stride_v_d

        # Load K, V blocks
        kv_mask = mask_t[:, None] & (offs_d[None, :] < D)
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)  # [BLOCK_T, BLOCK_D]
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)  # [BLOCK_T, BLOCK_D]

        # Compute attention scores: q @ K^T
        scores = tl.sum(k * q[None, :], axis=1) * scale  # [BLOCK_T]
        scores = tl.where(mask_t, scores, neg_inf)

        # Online-softmax update (fp32)
        block_max = tl.max(scores, axis=0)
        new_max   = tl.maximum(max_score, block_max)
        rescale   = tl.where(max_score == neg_inf, 0.0, tl.exp(max_score - new_max))
        acc       *= rescale
        sum_exp   *= rescale

        # Probabilities
        safe_scores = tl.where(mask_t, scores, new_max)
        probs = tl.exp(safe_scores - new_max)
        probs = tl.where(mask_t, probs, 0.0)

        # Update accumulator and sum
        acc       += tl.sum(probs[:, None] * v, axis=0)  # [BLOCK_D]
        sum_exp   += tl.sum(probs, axis=0)
        max_score = new_max

    # Store partition outputs
    m_ptr = M + off_b*stride_m_b + off_h*stride_m_h + pid_p*stride_m_p
    l_ptr = L + off_b*stride_l_b + off_h*stride_l_h + pid_p*stride_l_p
    a_ptr = A + off_b*stride_a_b + off_h*stride_a_h + pid_p*stride_a_p + offs_d*stride_a_d
    tl.store(m_ptr, tl.where(is_empty, neg_inf, max_score))
    tl.store(l_ptr, tl.where(is_empty, 0.0,     sum_exp))
    tl.store(a_ptr, tl.where(offs_d < D, tl.where(is_empty, 0.0, acc), 0.0))


@triton.jit
def flash_decoding_2_pass1(
    Q,      # [B, H, 1, D] (bf16)
    K, V,   # [B, H, T, D] (bf16)
    M,      # [B, H, P] (fp32)    partition max
    L,      # [B, H, P] (fp32)    partition sum-exp
    A,      # [B, H, P, D] (fp32) partition accumulator over D
    stride_q_b, stride_q_h, stride_q_t, stride_q_d,
    stride_k_b, stride_k_h, stride_k_t, stride_k_d,
    stride_v_b, stride_v_h, stride_v_t, stride_v_d,
    stride_m_b, stride_m_h, stride_m_p,
    stride_l_b, stride_l_h, stride_l_p,
    stride_a_b, stride_a_h, stride_a_p, stride_a_d,
    B, H, T, D,
    FIXMAX: tl.constexpr,  # partition-wise fixed max
    T_PART: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # Program IDs
    pid_hb = tl.program_id(0)
    pid_p  = tl.program_id(1)
    off_h  = pid_hb % H
    off_b  = pid_hb // H

    # Offsets
    offs_d = tl.arange(0, BLOCK_D)
    offs_t = tl.arange(0, BLOCK_T)

    # Pointers
    q_ptrs = Q + off_b*stride_q_b + off_h*stride_q_h + 0*stride_q_t + offs_d*stride_q_d
    k_base = K + off_b*stride_k_b + off_h*stride_k_h
    v_base = V + off_b*stride_v_b + off_h*stride_v_h

    # Load query vector
    q = tl.load(q_ptrs, mask=offs_d < D, other=0.0).to(tl.float32)

    # partition bounds
    t_start  = pid_p * T_PART
    t_end    = tl.minimum(t_start + T_PART, T)
    is_empty = t_start >= T

    # constants
    Df    = tl.full((), 0.0, dtype=tl.float32) + D
    scale = 1.0 / tl.sqrt(Df)
    neg_inf = tl.full((), -float("inf"), dtype=tl.float32)

    # Fix-max softmax
    fixmax = tl.full((), FIXMAX, dtype=tl.float32)
    sum_exp    = tl.zeros((), dtype=tl.float32)
    acc        = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Loop over K, V blocks along T-partition
    for bidx in range(0, N_BLOCKS):
        tile_t = t_start + bidx*BLOCK_T + offs_t
        mask_t = (tile_t < t_end) & (~is_empty)

        # Pointers
        k_ptrs = k_base + tile_t[:, None]*stride_k_t + offs_d[None, :]*stride_k_d
        v_ptrs = v_base + tile_t[:, None]*stride_v_t + offs_d[None, :]*stride_v_d

        # Load K, V blocks
        kv_mask = mask_t[:, None] & (offs_d[None, :] < D)
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)  # [BLOCK_T, BLOCK_D]
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)  # [BLOCK_T, BLOCK_D]

        # Compute attention scores: q @ K^T
        scores = tl.sum(k * q[None, :], axis=1) * scale               # [BLOCK_T]
        safe_scores = tl.where(mask_t, scores, fixmax)

        # Probabilities
        probs = tl.exp(safe_scores - fixmax)
        probs = tl.where(mask_t, probs, 0.0)

        # Update accumulator and sum
        acc     += tl.sum(probs[:, None] * v, axis=0)
        sum_exp += tl.sum(probs, axis=0)

    # Store partition outputs
    m_ptr = M + off_b * stride_m_b + off_h * stride_m_h + pid_p * stride_m_p
    l_ptr = L + off_b * stride_l_b + off_h * stride_l_h + pid_p * stride_l_p
    a_ptr = A + off_b * stride_a_b + off_h * stride_a_h + pid_p * stride_a_p + offs_d * stride_a_d
    tl.store(m_ptr, tl.where(is_empty, neg_inf, fixmax))  # m = fixed max
    tl.store(l_ptr, tl.where(is_empty, 0.0, sum_exp))  # l = sum exp(score - mfix)
    tl.store(a_ptr, tl.where(offs_d < D, tl.where(is_empty, 0.0, acc), 0.0))

@triton.jit
def flash_decoding_pass2(
    M, L, A,  # partials
    O,        # [B, H, 1, D]
    stride_m_b, stride_m_h, stride_m_p,
    stride_l_b, stride_l_h, stride_l_p,
    stride_a_b, stride_a_h, stride_a_p, stride_a_d,
    stride_o_b, stride_o_h, stride_o_t, stride_o_d,
    B, H,
    P: tl.constexpr, D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Program IDs
    pid_hb = tl.program_id(0)
    off_h  = pid_hb % H
    off_b  = pid_hb // H

    # Offsets
    offs_p = tl.arange(0, P)

    # Pointers
    m_ptrs = M + off_b*stride_m_b + off_h*stride_m_h + offs_p*stride_m_p
    l_ptrs = L + off_b*stride_l_b + off_h*stride_l_h + offs_p*stride_l_p
    m_base = M + off_b * stride_m_b + off_h * stride_m_h
    a_base = A + off_b * stride_a_b + off_h * stride_a_h

    # Load all partition stats
    m_i = tl.load(m_ptrs, mask=offs_p < P, other=-float("inf"))  # [P]
    l_i = tl.load(l_ptrs, mask=offs_p < P, other=0.0)            # [P]
    m   = tl.max(m_i, axis=0)                                    # scalar

    # Combine sums: Lg = Σ l_i * exp(m_i - m)
    w   = tl.exp(m_i - m)                                        # [P]
    Lg  = tl.sum(l_i * w, axis=0)                                # scalar

    # Combine accumulators: A = Σ A_i * exp(m_i - m)
    offs_d = tl.arange(0, BLOCK_D)
    a_acc  = tl.zeros([BLOCK_D], dtype=tl.float32)

    # static loop
    for p in tl.static_range(0, P):
        a_ptrs = a_base + p * stride_a_p + offs_d * stride_a_d
        a_i = tl.load(a_ptrs, mask=offs_d < D, other=0.0)   # [BLOCK_D]
        m_ip = tl.load(m_base + p * stride_m_p)
        wp = tl.exp(m_ip - m)
        a_acc += a_i * wp

    # Final normalization
    out = a_acc / tl.maximum(Lg, 1e-20)

    # Store output
    o_ptrs = O + off_b * stride_o_b + off_h * stride_o_h + 0 * stride_o_t + offs_d * stride_o_d
    tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=offs_d < D)

def flash_decoding(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """
    Flash Decoding Triton Kernel

    Args:
        q: Query tensor [batch, n_heads, 1, d_head]
        k: Key tensor   [batch, n_heads, seq_len_k, d_head]
        v: Value tensor [batch, n_heads, seq_len_v, d_head]

    Returns:
        o: Output tensor [batch, n_heads, 1, d_head]
    """
    _, _, q_len, _ = q.shape
    batch_size, n_heads, kv_seq_len, d_heads = k.shape
    assert q.ndim == k.ndim == v.ndim == 4
    assert q_len == 1

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    num_kv_split = NUM_KV_SPLIT

    m = torch.empty(
        (batch_size, n_heads, num_kv_split),
        device=q.device, dtype=torch.float32
    )
    l = torch.empty(
        (batch_size, n_heads, num_kv_split),
        device=q.device, dtype=torch.float32
    )
    a = torch.empty(
        (batch_size, n_heads, num_kv_split, d_heads),
        device=q.device, dtype=torch.float32
    )
    o = torch.empty_like(q)

    t_part = triton.cdiv(kv_seq_len, num_kv_split)
    num_blocks = triton.cdiv(t_part, BLOCK_T)

    # PASS 1
    grid1 = (batch_size * n_heads, num_kv_split)
    flash_decoding_pass1[grid1](
        q, k, v,
        m, l, a,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        m.stride(0), m.stride(1), m.stride(2),
        l.stride(0), l.stride(1), l.stride(2),
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        batch_size, n_heads, kv_seq_len, d_heads,
        T_PART=t_part,
        N_BLOCKS=num_blocks,
        BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES
    )

    # PASS 2
    grid2 = (batch_size * n_heads, )
    flash_decoding_pass2[grid2](
        m, l, a,
        o,
        m.stride(0), m.stride(1), m.stride(2),
        l.stride(0), l.stride(1), l.stride(2),
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        batch_size, n_heads, num_kv_split, d_heads,
        BLOCK_D=BLOCK_D,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES
    )
    return o

def flash_decoding_2(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    fix_max: float = 10,
) -> torch.Tensor:
    """
        Flash Decoding 2 Triton Kernel (FIX-MAX SOFTMAX)

        Args:
            q: Query tensor [batch, n_heads, 1, d_head]
            k: Key tensor   [batch, n_heads, seq_len_k, d_head]
            v: Value tensor [batch, n_heads, seq_len_v, d_head]
            fix_max: scalar

        Returns:
            o: Output tensor [batch, n_heads, 1, d_head]
        """
    _, _, q_len, _ = q.shape
    batch_size, n_heads, kv_seq_len, d_heads = k.shape
    assert q.ndim == k.ndim == v.ndim == 4
    assert q_len == 1

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    num_kv_split = NUM_KV_SPLIT

    m = torch.empty(
        (batch_size, n_heads, num_kv_split),
        device=q.device, dtype=torch.float32
    )
    l = torch.empty(
        (batch_size, n_heads, num_kv_split),
        device=q.device, dtype=torch.float32
    )
    a = torch.empty(
        (batch_size, n_heads, num_kv_split, d_heads),
        device=q.device, dtype=torch.float32
    )
    o = torch.empty_like(q)

    t_part = triton.cdiv(kv_seq_len, num_kv_split)
    num_blocks = triton.cdiv(t_part, BLOCK_T)

    # PASS 1
    grid1 = (batch_size * n_heads, num_kv_split)
    flash_decoding_2_pass1[grid1](
        q, k, v,
        m, l, a,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        m.stride(0), m.stride(1), m.stride(2),
        l.stride(0), l.stride(1), l.stride(2),
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        batch_size, n_heads, kv_seq_len, d_heads,
        fix_max,
        T_PART=t_part,
        N_BLOCKS=num_blocks,
        BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES
    )

    # PASS 2
    grid2 = (batch_size * n_heads,)
    flash_decoding_pass2[grid2](
        m, l, a,
        o,
        m.stride(0), m.stride(1), m.stride(2),
        l.stride(0), l.stride(1), l.stride(2),
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        batch_size, n_heads, num_kv_split, d_heads,
        BLOCK_D=BLOCK_D,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES
    )
    return o


def estimate_memory(batch_size: int, n_heads: int, kv_seq_len: int, d_head: int, dtype: torch.dtype):
    if dtype == torch.float16 or dtype == torch.bfloat16:
        bytes_per_element = 2
    elif dtype == torch.float32:
        bytes_per_element = 4
    else:
        raise ValueError("Unsupported data type")
    total_bytes = batch_size * n_heads * kv_seq_len * d_head * 2 * bytes_per_element # 2 -> Keys & Values
    memory = total_bytes / (1024 ** 3)
    return memory

def speedometer(fn, *args, num_warmups: int, num_runs: int):
    output = None

    for _ in range(num_warmups):
        _ = fn(*args)

    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_runs):
        output = fn(*args)

    torch.cuda.synchronize()
    end_time = time.time()

    return (end_time - start_time) / num_runs, output

def benchmark(
    configs, algorithms,
    device: torch.device, dtype: torch.dtype, vram_size: int,
    num_warmup: int = 5, num_run: int = 10,
    rtol: float = 1e-2, atol: float = 1e-2
):
    print("\n\033[95m" + "═" * 75)
    print("🚀 Flash Attention - Benchmark Results")

    for config in configs:
        batch_size = config["batch_size"]
        n_heads = config["n_heads"]
        d_head = config["d_head"]
        kv_seq_len = config["kv_seq_len"]

        memory = estimate_memory(batch_size, n_heads, kv_seq_len, d_head, dtype=dtype)
        if memory > vram_size:
            print("\033[95m" + "═" * 75 + "\033[0m")
            print("\033[95m🧩 Configuration\033[0m")
            print(f"  • Batch Size : {batch_size}")
            print(f"  • Num Heads  : {n_heads}")
            print(f"  • Head Dim   : {d_head}")
            print(f"  • KV Seq Len : {kv_seq_len}")
            print(f"  • KV Cache   : {memory:.2f} GB")
            print(f"  Memory limit exceeded: {memory:.2f} GB > {vram_size:.2f} GB\n")
            continue

        try:
            q = torch.randn(batch_size, n_heads, 1, d_head, device=device, dtype=dtype)
            k = torch.randn(batch_size, n_heads, kv_seq_len, d_head, device=device, dtype=dtype)
            v = torch.randn(batch_size, n_heads, kv_seq_len, d_head, device=device, dtype=dtype)

            results = []
            for name, fn in algorithms:
                try:
                    exec_time, output = speedometer(fn, q, k, v, num_warmups=num_warmup, num_runs=num_run)
                    results.append((name, exec_time, output))
                except Exception as e:
                    print(f"{name} failed: {e}")

            print("\033[95m" + "═" * 75 + "\033[0m")
            print(f"\033[94m| {'Algorithm':<20} | {'Latency':>10} | {'Speedup':>8} | {'Throughput':>12} | {'Accuracy':>12} |\033[0m")
            print(f"|{'-' * 22}|{'-' * 12}|{'-' * 10}|{'-' * 14}|{'-' * 14}|")

            if len(results) == 0:
                print("| (no results)".ljust(74) + "|")
                print(f"|{'-' * 22}|{'-' * 12}|{'-' * 10}|{'-' * 14}|{'-' * 14}|")
            else:
                baseline_time = results[0][1]
                baseline_output = results[0][2]

                for name, exec_time, output in results:
                    if exec_time and exec_time > 0:
                        speedup_val = (baseline_time / exec_time) if baseline_time else float('inf')
                        throughput_val = batch_size / exec_time / 1e3
                    else:
                        speedup_val = float('inf')
                        throughput_val = float('inf')

                    speedup = f"{speedup_val:.2f}x" if speedup_val != float('inf') else "inf"
                    throughput = f"{throughput_val:.2f}" if throughput_val != float('inf') else "inf"

                    try:
                        ok = torch.allclose(output, baseline_output, rtol=rtol, atol=atol)
                        if ok:
                            accuracy = "\033[92m ✅ Pass   \033[0m"  # ← 공백 맞춤
                        else:
                            max_err = torch.max(torch.abs(output - baseline_output)).item()
                            accuracy = f"\033[93m ⚠️ {max_err:.1e}\033[0m"
                    except Exception as e:
                        accuracy = f"\033[91m ⚠️ {str(e).split(':')[0]}\033[0m"

                    print(f"| \033[96m{name:<20}\033[0m | {exec_time * 1000:>10.2f} | {speedup:>8} | {throughput:>12} | {accuracy:<12} |")

                print(f"|{'-' * 22}|{'-' * 12}|{'-' * 10}|{'-' * 14}|{'-' * 14}|")

            print("\033[95m🧩 Configuration\033[0m")
            print(f"  • Batch Size : {batch_size}")
            print(f"  • Num Heads  : {n_heads}")
            print(f"  • Head Dim   : {d_head}")
            print(f"  • KV Seq Len : {kv_seq_len}")
            print(f"  • KV Cache   : {memory:.2f} GB\n")

        except Exception as e:
            print(f"Unexpected error in config {config}: {e}")


def main():
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    dtype = torch.float32
    properties = driver.active.utils.get_device_properties(driver.active.get_current_device())
    num_sm = properties["multiprocessor_count"]
    num_regs = properties["max_num_regs"]
    sram_size = properties["max_shared_mem"]
    warp_size = properties["warpSize"]
    vram_size: int = 16
    print(f"Device: {torch.cuda.get_device_name(device)}\n"
          f"Number of SM: {num_sm}\n"
          f"Number of registers: {num_regs}\n"
          f"Size of SMEM: {sram_size}\n"
          f"Warp size: {warp_size}")
    set_seed(42)

    # batch_sizes = [1, 4, 32]
    # n_heads     = [12, 16, 16, 24]
    # d_heads     = [64, 64, 96, 128]
    # kv_seq_lens = [1024, 2048, 4096, 8192]
    test_configs = [
        # Single batch
        {"batch_size": 1, "n_heads": 12, "d_head": 64, "kv_seq_len": 1024},
        {"batch_size": 1, "n_heads": 16, "d_head": 64, "kv_seq_len": 2048},
        {"batch_size": 1, "n_heads": 16, "d_head": 96, "kv_seq_len": 4096},
        {"batch_size": 1, "n_heads": 24, "d_head": 128, "kv_seq_len": 8192},
        # Small batch
        {"batch_size": 4, "n_heads": 12, "d_head": 64, "kv_seq_len": 1024},
        {"batch_size": 4, "n_heads": 16, "d_head": 64, "kv_seq_len": 2048},
        {"batch_size": 4, "n_heads": 16, "d_head": 96, "kv_seq_len": 4096},
        {"batch_size": 4, "n_heads": 24, "d_head": 128, "kv_seq_len": 8192},
        # Large batch
        {"batch_size": 16, "n_heads": 12, "d_head": 64, "kv_seq_len": 1024},
        {"batch_size": 16, "n_heads": 16, "d_head": 64, "kv_seq_len": 2048},
        {"batch_size": 16, "n_heads": 16, "d_head": 86, "kv_seq_len": 4096},
        {"batch_size": 16, "n_heads": 24, "d_head": 128, "kv_seq_len": 8192},
        # Long Context
        {"batch_size": 1, "n_heads": 12, "d_head": 64, "kv_seq_len": 131072},
        {"batch_size": 2, "n_heads": 12, "d_head": 64, "kv_seq_len": 131072},
    ]
    algorithms = [
        ("Naive Attention", naive_attention),
        # ("Flash Attention (Triton)", flash_attn_1),
        ("FA2", F.scaled_dot_product_attention),
        ("FA2 (Triton)", flash_attn_2),
        ("FA2 + fix-max", flash_attn_2_fixmax),
        ("Flash Decoding", flash_decoding),
        ("Flash Decoding 2",flash_decoding_2),
    ]
    benchmark(
        configs=test_configs, algorithms=algorithms,
        device=device, dtype=dtype, vram_size=vram_size,
    )

if __name__ == "__main__":
    main()