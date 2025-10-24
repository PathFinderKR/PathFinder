import time
import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.runtime import driver
from src.utils import set_seed
MEMORY_LIMIT_GB = 16
BLOCK_N: tl.constexpr = 128
BLOCK_D: tl.constexpr = 64

def naive_attention(q, k, v):
    scale = 1.0 / math.sqrt(q.size(-1))
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_prob = torch.softmax(attn_scores, dim=-1)
    attn_out = torch.matmul(attn_prob, v)
    return attn_out

@triton.jit
def flash_attn_2_kernel(
        Q,    # [B, H, 1, D]
        K, V, # [B, H, T, D]
        O,    # [B, H, 1, D]
        stride_q_b, stride_q_h, stride_q_t, stride_q_d,
        stride_k_b, stride_k_h, stride_k_t, stride_k_d,
        stride_v_b, stride_v_h, stride_v_t, stride_v_d,
        stride_o_b, stride_o_h, stride_o_t,  stride_o_d,
        B, H, T, D,
        BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Program IDs
    off_hz = tl.program_id(0)
    off_h = off_hz % H
    off_z = off_hz // H

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # Pointers
    q_ptrs = Q + off_z * stride_q_b + off_h * stride_q_h + offs_d * stride_q_d
    k_ptrs = K + off_z * stride_k_b + off_h * stride_k_h + offs_n[:, None] * stride_k_t + offs_d[None, :] * stride_k_d
    v_ptrs = V + off_z * stride_v_b + off_h * stride_v_h + offs_n[:, None] * stride_v_t + offs_d[None, :] * stride_v_d
    o_ptrs = O + off_z * stride_o_b + off_h * stride_o_h + offs_d * stride_o_d

    # Load single query vector
    q = tl.load(q_ptrs, mask=offs_d < D, other=0.0).to(tl.float32)  # [BLOCK_D]

    # accumulator and softmax statistics
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    max_score = tl.full((), -float("inf"), tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    # Scale
    scale = 1.0 / tl.sqrt(tl.full((), D, tl.float32))

    # Loop over K, V blocks
    start_n = 0
    while start_n < T:
        # Calculate current block bounds
        remain = T - start_n
        block_size = tl.minimum(remain, BLOCK_N)
        mask_n = offs_n < block_size  # [BLOCK_N]

        # Load K, V blocks
        k = tl.load(
            k_ptrs + start_n * stride_k_t,
            mask=mask_n[:, None] & (offs_d[None, :] < D),
            other=0.0,
        )  # [BLOCK_N, BLOCK_D]
        v = tl.load(
            v_ptrs + start_n * stride_v_t,
            mask=mask_n[:, None] & (offs_d[None, :] < D),
            other=0.0,
        )  # [BLOCK_N, BLOCK_D]

        # Compute attention scores: q @ K^T
        scores = tl.sum(k * q[None, :], axis=1) * scale  # [BLOCK_N]
        neg_inf = tl.full(scores.shape, -float("inf"), scores.dtype)
        scores = tl.where(mask_n, scores, neg_inf)

        # Online softmax update
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(max_score, block_max)
        rescale = tl.exp(max_score - new_max)
        acc *= rescale
        sum_exp *= rescale

        # Compute probabilities for current block
        probs = tl.exp(scores - new_max)  # [BLOCK_N]
        probs = tl.where(mask_n, probs, 0.0)

        # Update accumulator and sum
        acc += tl.sum(probs[:, None] * v, axis=0)
        sum_exp += tl.sum(probs, axis=0)
        max_score = new_max

        start_n += BLOCK_N

    # Final normalization
    eps = tl.full((), 1e-20, tl.float32)
    denom = tl.maximum(sum_exp, eps)
    out = acc / denom

    # Store output
    tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=offs_d < D)

def flash_attn_2(q, k, v):
    """
    Flash Attention 2 Triton Kernel

    Args:
        q: Query tensor [batch, n_heads, 1, d_head]
        k: Key tensor   [batch, n_heads, seq_len_k, d_head]
        v: Value tensor [batch, n_heads, seq_len_v, d_head]

    Returns:
        o: Output tensor [batch, n_heads, 1, d_head]
    """
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    batch_size, n_heads, q_seq_len, d_head = q.shape
    assert q_seq_len == 1, "This kernel assumes q_len=1 (decode step)."
    kv_seq_len = k.shape[2]

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    o = torch.empty_like(q)
    grid = (batch_size * n_heads,)

    flash_attn_2_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  # Q strides [B, H, 1, D]
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  # K strides [B, H, N, D]
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),  # V strides [B, H, N, D]
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),  # O strides [B, H, 1, D]
        batch_size, n_heads, kv_seq_len, d_head,
        BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2
    )
    return o

@triton.jit
def flash_decoding_stage_1(
    Q, K, V,
    FIXMAX, ACC, SUM,
    stride_q_b, stride_q_h, stride_q_t, stride_q_d,
    stride_k_b, stride_k_h, stride_k_t, stride_k_d,
    stride_v_b, stride_v_h, stride_v_t, stride_v_d,
    stride_m_b, stride_m_h,
    stride_acc_b, stride_acc_h, stride_acc_d,
    stride_sum_b, stride_sum_h,
    B, H, T, D,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_bh = tl.program_id(0)
    pid_t  = tl.program_id(1)

    off_h = pid_bh % H
    off_z = pid_bh // H

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q + off_z*stride_q_b + off_h*stride_q_h + offs_d*stride_q_d
    k_base = K + off_z*stride_k_b + off_h*stride_k_h
    v_base = V + off_z*stride_v_b + off_h*stride_v_h

    # fixed max
    max_ptr = FIXMAX + off_z*stride_m_b + off_h*stride_m_h
    fixed_max = tl.load(max_ptr).to(tl.float32)

    # tile range on T
    start_n = pid_t * BLOCK_N
    remain = T - start_n
    block_size = tl.maximum(0, tl.minimum(remain, BLOCK_N))
    mask_n = offs_n < block_size

    # load q (fp32)
    q = tl.load(q_ptrs, mask=offs_d < D, other=0.0).to(tl.float32)

    # scale
    scale = 1.0 / tl.sqrt(tl.full((), D, tl.float32))

    # partial accumulators
    part_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    part_sum = tl.zeros((), dtype=tl.float32)

    # load k,v tile (fp32)
    k = tl.load(
        k_base + (start_n + offs_n)[:, None]*stride_k_t + offs_d[None, :]*stride_k_d,
        mask=mask_n[:, None] & (offs_d[None, :] < D),
        other=0.0,
    ).to(tl.float32)
    v = tl.load(
        v_base + (start_n + offs_n)[:, None]*stride_v_t + offs_d[None, :]*stride_v_d,
        mask=mask_n[:, None] & (offs_d[None, :] < D),
        other=0.0,
    ).to(tl.float32)

    # scores
    scores = tl.sum(k * q[None, :], axis=1) * scale
    neg_inf = tl.full(scores.shape, -float("inf"), scores.dtype)
    scores = tl.where(mask_n, scores, neg_inf)

    # probs with fixed max
    probs = tl.exp(scores - fixed_max)
    probs = tl.where(mask_n, probs, 0.0)

    # partial sums
    part_sum += tl.sum(probs, axis=0)
    part_acc += tl.sum(probs[:, None] * v, axis=0)

    # atomic accumulate into ACC[B,H,D] and SUM[B,H]
    acc_ptrs = ACC + off_z*stride_acc_b + off_h*stride_acc_h + offs_d*stride_acc_d
    sum_ptr  = SUM + off_z*stride_sum_b + off_h*stride_sum_h

    # only valid D range
    tl.atomic_add(acc_ptrs, part_acc, mask=offs_d < D)
    tl.atomic_add(sum_ptr,  part_sum)

@triton.jit
def flash_decoding_stage_2(
    ACC, SUM,
    O,
    stride_acc_b, stride_acc_h, stride_acc_d,
    stride_sum_b, stride_sum_h,
    stride_o_b, stride_o_h, stride_o_t, stride_o_d,
    B, H, D,
    BLOCK_D: tl.constexpr
):
    pid_bh = tl.program_id(0)
    off_h = pid_bh % H
    off_z = pid_bh // H

    offs_d = tl.arange(0, BLOCK_D)

    acc_ptrs = ACC + off_z*stride_acc_b + off_h*stride_acc_h + offs_d*stride_acc_d
    sum_ptr  = SUM + off_z*stride_sum_b + off_h*stride_sum_h
    o_ptrs   = O   + off_z*stride_o_b   + off_h*stride_o_h   + offs_d*stride_o_d  # t=0

    acc = tl.load(acc_ptrs, mask=offs_d < D, other=0.0)
    s   = tl.load(sum_ptr)
    eps = tl.full((), 1e-20, tl.float32)
    denom = tl.maximum(s, eps)
    out = acc / denom

    tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=offs_d < D)

def flash_decoding_2(q, k, v, fixmax: float = 10):
    B, H, _, D = q.shape
    T = k.shape[2]
    n_tiles: int = (T + BLOCK_N - 1) // BLOCK_N

    ACC = torch.zeros((B, H, D), dtype=torch.float32, device=q.device)
    SUM = torch.zeros((B, H),    dtype=torch.float32, device=q.device)

    grid_partial = (B*H, n_tiles)
    flash_decoding_stage_1[grid_partial](
        q, k, v, fixmax, ACC, SUM,
        # strides ... (ACC/SUM 포함),
        B, H, T, D,
        BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2
    )

    o = torch.empty_like(q)
    grid_finalize = (B*H,)
    flash_decoding_stage_2[grid_finalize](
        ACC, SUM, o,
        # strides ...,
        B, H, D,
        BLOCK_D=BLOCK_D,
        num_warps=1
    )
    return o

@triton.jit
def flash_decoding_2_kernel(
        Q,     # [B, H, 1, D]
        K, V,  # [B, H, T, D]
        O,     # [B, H, 1, D]
        FIXMAX,
        stride_q_b, stride_q_h, stride_q_t, stride_q_d,
        stride_k_b, stride_k_h, stride_k_t, stride_k_d,
        stride_v_b, stride_v_h, stride_v_t, stride_v_d,
        stride_o_b, stride_o_h, stride_o_t, stride_o_d,
        stride_m_b, stride_m_h,
        B, H, T, D,
        BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Program IDs
    off_hz = tl.program_id(0)
    off_h = off_hz % H
    off_z = off_hz // H

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # Initialize pointers for this batch and head
    q_ptrs = Q + off_z * stride_q_b + off_h * stride_q_h + offs_d * stride_q_d
    k_ptrs = K + off_z * stride_k_b + off_h * stride_k_h + offs_n[:, None] * stride_k_t + offs_d[None, :] * stride_k_d
    v_ptrs = V + off_z * stride_v_b + off_h * stride_v_h + offs_n[:, None] * stride_v_t + offs_d[None, :] * stride_v_d
    o_ptrs = O + off_z * stride_o_b + off_h * stride_o_h + offs_d * stride_o_d

    #
    max_ptr = FIXMAX + off_z * stride_m_b + off_h * stride_m_h
    fixed_max = tl.load(max_ptr, mask=tl.full((), True, tl.int1), other=0.0).to(tl.float32)

    # Load single query vector
    q = tl.load(q_ptrs, mask=offs_d < D, other=0.0).to(tl.float32)  # [BLOCK_D]

    # Initialize output accumulator and softmax statistics
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    # Scale
    scale = 1.0 / tl.sqrt(tl.full((), D, tl.float32))

    # Loop over K, V blocks
    start_n = 0
    while start_n < T:
        # Calculate current block bounds
        remain = T - start_n
        block_size = tl.minimum(remain, BLOCK_N)
        mask_n = offs_n < block_size

        # Load K, V blocks
        k = tl.load(
            k_ptrs + start_n * stride_k_t,
            mask=mask_n[:, None] & (offs_d[None, :] < D),
            other=0.0,
        )  # [BLOCK_N, BLOCK_D]
        v = tl.load(
            v_ptrs + start_n * stride_v_t,
            mask=mask_n[:, None] & (offs_d[None, :] < D),
            other=0.0,
        )  # [BLOCK_N, BLOCK_D]

        # Compute attention scores: q @ K^T
        scores = tl.sum(k * q[None, :], axis=1) * scale  # [BLOCK_N]

        # Apply causal mask if needed (for decoding, usually all positions are valid)
        scores = tl.where(mask_n, scores, -float('inf'))
        neg_inf = tl.full(scores.shape, -float("inf"), scores.dtype)
        scores = tl.where(mask_n, scores, neg_inf)

        # Compute probabilities for current block
        probs = tl.exp(scores - fixed_max)  # [BLOCK_N]
        probs = tl.where(mask_n, probs, 0.0)

        # Update accumulator and sum
        sum_exp += tl.sum(probs, axis=0)
        acc += tl.sum(probs[:, None] * v, axis=0)  # [BLOCK_D]

        start_n += BLOCK_N

    # Final normalization
    eps = tl.full((), 1e-20, tl.float32)
    denom = tl.maximum(sum_exp, eps)
    out = acc / denom

    # Store output
    tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=offs_d < D)

def flash_decoding_2(q, k, v, fixmax: float = 10):
    """
    Flash Decoding 2 Triton Kernel

    Args:
        q: Query tensor [batch, n_heads, 1, d_head]
        k: Key tensor   [batch, n_heads, seq_len_k, d_head]
        v: Value tensor [batch, n_heads, seq_len_v, d_head]
        fixmax:

    Returns:
        o: Output tensor [batch, n_heads, 1, d_head]
    """
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    batch_size, n_heads, q_seq_len, d_head = q.shape
    assert q_seq_len == 1, "This kernel assumes q_len=1 (decode step)."
    kv_seq_len = k.shape[2]

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    fixmax = torch.full((batch_size, n_heads), float(fixmax), dtype=torch.float32, device=q.device)

    o = torch.empty_like(q)
    grid = (batch_size * n_heads,)

    flash_decoding_2_kernel[grid](
        q, k, v, o,
        fixmax,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  # Q strides [B, H, 1, D]
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  # K strides [B, H, N, D]
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),  # V strides [B, H, N, D]
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),  # O strides [B, H, 1, D]
        fixmax.stride(0), fixmax.stride(1),
        batch_size, n_heads, kv_seq_len, d_head,
        BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2
    )
    return o

@triton.jit
def flash_decoding_2_kernel(
        Q,     # [B, H, 1, D]
        K, V,  # [B, H, T, D]
        O,     # [B, H, 1, D]
        FIXMAX,
        stride_q_b, stride_q_h, stride_q_t, stride_q_d,
        stride_k_b, stride_k_h, stride_k_t, stride_k_d,
        stride_v_b, stride_v_h, stride_v_t, stride_v_d,
        stride_o_b, stride_o_h, stride_o_t, stride_o_d,
        stride_m_b, stride_m_h,
        B, H, T, D,
        BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Program IDs
    off_hz = tl.program_id(0)
    off_h = off_hz % H
    off_z = off_hz // H

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # Initialize pointers for this batch and head
    q_ptrs = Q + off_z * stride_q_b + off_h * stride_q_h + offs_d * stride_q_d
    k_ptrs = K + off_z * stride_k_b + off_h * stride_k_h + offs_n[:, None] * stride_k_t + offs_d[None, :] * stride_k_d
    v_ptrs = V + off_z * stride_v_b + off_h * stride_v_h + offs_n[:, None] * stride_v_t + offs_d[None, :] * stride_v_d
    o_ptrs = O + off_z * stride_o_b + off_h * stride_o_h + offs_d * stride_o_d

    #
    max_ptr = FIXMAX + off_z * stride_m_b + off_h * stride_m_h
    fixed_max = tl.load(max_ptr, mask=tl.full((), True, tl.int1), other=0.0).to(tl.float32)

    # Load single query vector
    q = tl.load(q_ptrs, mask=offs_d < D, other=0.0).to(tl.float32)  # [BLOCK_D]

    # Initialize output accumulator and softmax statistics
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    # Scale
    scale = 1.0 / tl.sqrt(tl.full((), D, tl.float32))

    # Loop over K, V blocks
    start_n = 0
    while start_n < T:
        # Calculate current block bounds
        remain = T - start_n
        block_size = tl.minimum(remain, BLOCK_N)
        mask_n = offs_n < block_size

        # Load K, V blocks
        k = tl.load(
            k_ptrs + start_n * stride_k_t,
            mask=mask_n[:, None] & (offs_d[None, :] < D),
            other=0.0,
        )  # [BLOCK_N, BLOCK_D]
        v = tl.load(
            v_ptrs + start_n * stride_v_t,
            mask=mask_n[:, None] & (offs_d[None, :] < D),
            other=0.0,
        )  # [BLOCK_N, BLOCK_D]

        # Compute attention scores: q @ K^T
        scores = tl.sum(k * q[None, :], axis=1) * scale  # [BLOCK_N]

        # Apply causal mask if needed (for decoding, usually all positions are valid)
        scores = tl.where(mask_n, scores, -float('inf'))
        neg_inf = tl.full(scores.shape, -float("inf"), scores.dtype)
        scores = tl.where(mask_n, scores, neg_inf)

        # Compute probabilities for current block
        probs = tl.exp(scores - fixed_max)  # [BLOCK_N]
        probs = tl.where(mask_n, probs, 0.0)

        # Update accumulator and sum
        sum_exp += tl.sum(probs, axis=0)
        acc += tl.sum(probs[:, None] * v, axis=0)  # [BLOCK_D]

        start_n += BLOCK_N

    # Final normalization
    eps = tl.full((), 1e-20, tl.float32)
    denom = tl.maximum(sum_exp, eps)
    out = acc / denom

    # Store output
    tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=offs_d < D)

def flash_decoding_2(q, k, v, fixmax: float = 10):
    """
    Flash Decoding 2 Triton Kernel

    Args:
        q: Query tensor [batch, n_heads, 1, d_head]
        k: Key tensor   [batch, n_heads, seq_len_k, d_head]
        v: Value tensor [batch, n_heads, seq_len_v, d_head]
        fixmax:

    Returns:
        o: Output tensor [batch, n_heads, 1, d_head]
    """
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    batch_size, n_heads, q_seq_len, d_head = q.shape
    assert q_seq_len == 1, "This kernel assumes q_len=1 (decode step)."
    kv_seq_len = k.shape[2]

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    fixmax = torch.full((batch_size, n_heads), float(fixmax), dtype=torch.float32, device=q.device)

    o = torch.empty_like(q)
    grid = (batch_size * n_heads,)

    flash_decoding_2_kernel[grid](
        q, k, v, o,
        fixmax,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  # Q strides [B, H, 1, D]
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  # K strides [B, H, N, D]
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),  # V strides [B, H, N, D]
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),  # O strides [B, H, 1, D]
        fixmax.stride(0), fixmax.stride(1),
        batch_size, n_heads, kv_seq_len, d_head,
        BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2
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

def print_results(algorithms, batch_size, n_heads, d_head, kv_seq_len, memory, rtol, atol):
    print(f"\n{'Algorithm':<20} {'Time(ms)':<10} {'Speedup':<10} {'Throughput(M tok/s)':<20} {'Accuracy':<10}")
    print("─" * 80)

    baseline_time = algorithms[0][1]
    baseline_output = algorithms[0][2]

    for name, exec_time, output in algorithms:
        if exec_time and exec_time > 0:
            speedup_val = (baseline_time / exec_time) if baseline_time else float('inf')
            throughput_val = (batch_size * kv_seq_len) / exec_time / 1e6
        else:
            speedup_val = float('inf')
            throughput_val = float('inf')

        speedup = f"{speedup_val:.2f}x" if speedup_val != float('inf') else "inf"
        throughput = f"{throughput_val:.2f}" if throughput_val != float('inf') else "inf"

        try:
            ok = torch.allclose(output, baseline_output, rtol=rtol, atol=atol)
            if ok:
                accuracy = "✅ Pass"
            else:
                max_err = torch.max(torch.abs(output - baseline_output)).item()
                accuracy = f"⚠️ {max_err:.1e}"
        except Exception as e:
            accuracy = f"⚠️ compare error: {e}"

        print(f"{name:<20} {exec_time * 1000:<10.2f} {speedup:<10} {throughput:<20} {accuracy:<10}")

    print(f"\nConfiguration:")
    print(f"  batch_size={batch_size}, n_heads={n_heads}, d_head={d_head}, kv_seq_len={kv_seq_len}")
    print(f"  KV Cache={memory:.2f} GB\n")

def benchmark(
    configs, algorithms,
    dtype: torch.dtype, device: torch.device,
    num_warmup: int = 5, num_run: int = 10,
    rtol: float = 1e-3, atol: float = 1e-3
):
    all_results = []

    for config in configs:
        batch_size = config["batch_size"]
        n_heads = config["n_heads"]
        d_head = config["d_head"]
        kv_seq_len = config["kv_seq_len"]

        memory = estimate_memory(batch_size, n_heads, kv_seq_len, d_head)
        if memory > MEMORY_LIMIT_GB:
            print(f"\nConfiguration: {config}")
            print(f"  Memory limit exceeded: {memory:.2f} GB > {MEMORY_LIMIT_GB:.2f} GB")
            continue

        try:
            q = torch.randn(batch_size, n_heads, 1, d_head, device=q.device, dtype=dtype)
            k = torch.randn(batch_size, n_heads, kv_seq_len, d_head, device=device, dtype=dtype)
            v = torch.randn(batch_size, n_heads, kv_seq_len, d_head, device=device, dtype=dtype)

            results = []
            for name, fn in algorithms:
                try:
                    exec_time, output = speedometer(fn, q, k, v, num_warmups=num_warmup, num_runs=num_run)
                    results.append((name, exec_time, output))
                except Exception as e:
                    print(f"{name} failed: {e}")

            print_results(results, batch_size, n_heads, d_head, kv_seq_len, memory, rtol, atol)
            all_results.append((config, results))

        except Exception as e:
            print(f"Unexpected error in config {config}: {e}")

    return all_results

def main():
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("highest")
    dtype = torch.float32
    DEVICE = driver.active.get_current_device()
    properties = driver.active.utils.get_device_properties(DEVICE)
    NUM_SM = properties["multiprocessor_count"]
    NUM_REGS = properties["max_num_regs"]
    SIZE_SMEM = properties["max_shared_mem"]
    WARP_SIZE = properties["warpSize"]
    print(f"Device: {torch.cuda.get_device_name(device)}\n"
          f"Number of SM: {NUM_SM}\n"
          f"Number of registers: {NUM_REGS}\n"
          f"Size of SMEM: {SIZE_SMEM}\n"
          f"Warp size: {WARP_SIZE}")
    set_seed(42)

    # batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    # n_heads = [12, 24, 32, 40, 96]
    # kv_seq_lens = [1024, 2048, 4096, 8192, 128000]
    # d_heads = [64, 80, 96, 128]
    test_configs = [
        # Single batch
        {"batch_size": 1, "n_heads": 12, "d_head": 64, "kv_seq_len": 1024},
        {"batch_size": 1, "n_heads": 24, "d_head": 80, "kv_seq_len": 2048},
        {"batch_size": 1, "n_heads": 32, "d_head": 96, "kv_seq_len": 4096},
        {"batch_size": 1, "n_heads": 40, "d_head": 128, "kv_seq_len": 8192},
        # Small batch
        {"batch_size": 4, "n_heads": 12, "d_head": 64, "kv_seq_len": 1024},
        {"batch_size": 4, "n_heads": 24, "d_head": 80, "kv_seq_len": 2048},
        {"batch_size": 4, "n_heads": 32, "d_head": 96, "kv_seq_len": 4096},
        {"batch_size": 4, "n_heads": 40, "d_head": 128, "kv_seq_len": 8192},
        # Large batch
        {"batch_size": 64, "n_heads": 12, "d_head": 64, "kv_seq_len": 1024},
        {"batch_size": 64, "n_heads": 24, "d_head": 80, "kv_seq_len": 2048},
        {"batch_size": 64, "n_heads": 32, "d_head": 96, "kv_seq_len": 4096},
        {"batch_size": 64, "n_heads": 40, "d_head": 128, "kv_seq_len": 8192},
        # Long Context
        {"batch_size": 1, "n_heads": 12, "d_head": 64, "kv_seq_len": 128000},
        {"batch_size": 2, "n_heads": 12, "d_head": 64, "kv_seq_len": 128000}
    ]
    algorithms = [
        ("Naive Attention", naive_attention),
        # ("Flash Attention (Triton)", flash_attn_1),
        ("Flash Attention 2", F.scaled_dot_product_attention),
        ("Flash Attention 2 (Triton)", flash_attn_2),
        ("Flash Attention 2 + fix-max", flash_attn_2_fixmax),
        ("Flash Decoding", flash_decoding),
        ("Flash Decoding 2",flash_decoding_2),
    ]
    benchmark(
        configs=test_configs, algorithms=algorithms,
        dtype=dtype, device=device
    )


if __name__ == "__main__":
    main()