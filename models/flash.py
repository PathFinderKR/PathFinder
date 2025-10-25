import math
import time
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.runtime import driver
from src.utils import set_seed
device = torch.device("cuda")
torch.set_float32_matmul_precision("high")
dtype = torch.float32
set_seed(42)
MEMORY_LIMIT_GB = 16
DEVICE = driver.active.get_current_device()
properties = driver.active.utils.get_device_properties(DEVICE)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
BLOCK_N = 64
BLOCK_D = 64

def print_device_info():
    print(f"Device: {torch.cuda.get_device_name(device)}\n"
          f"Number of SM: {NUM_SM}\n"
          f"Number of registers: {NUM_REGS}\n"
          f"Size of SMEM: {SIZE_SMEM}\n"
          f"Warp size: {WARP_SIZE}")

def estimate_memory(batch_size, n_heads, kv_seq_len, d_head):
    if dtype == torch.float16 or dtype == torch.bfloat16:
        bytes_per_element = 2
    elif dtype == torch.float32:
        bytes_per_element = 4
    else:
        raise ValueError("Unsupported data type")
    # 2 -> Keys & Values
    total_bytes = batch_size * n_heads * kv_seq_len * d_head * 2 * bytes_per_element
    memory = total_bytes / (1024 ** 3)
    return memory

def naive_attention(q, k, v, scale=None):
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_prob = torch.softmax(attn_scores, dim=-1)
    attn_out = torch.matmul(attn_prob, v)
    return attn_out

@triton.jit
def flash_attention_2_decode_kernel(
        Q, K, V, O,
        stride_qb, stride_qh, stride_qk,  # Q is [B, H, 1, D] - no seq dim stride needed
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_ok,  # O is [B, H, 1, D] - no seq dim stride needed
        B, H, N_CTX, D_HEAD,
        BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
):
    # Get program IDs - each program handles one (batch, head) pair
    off_hz = tl.program_id(0)
    off_h = off_hz % H
    off_z = off_hz // H

    # Compute offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # Initialize pointers for this batch and head
    # Q is shape [B, H, 1, D_HEAD] - single query token
    q_ptrs = Q + off_z * stride_qb + off_h * stride_qh + offs_d * stride_qk
    k_ptrs = K + off_z * stride_kb + off_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + off_z * stride_vb + off_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    o_ptrs = O + off_z * stride_ob + off_h * stride_oh + offs_d * stride_ok

    # Load single query vector
    q = tl.load(q_ptrs, mask=offs_d < D_HEAD, other=0.0)

    # Initialize output accumulator and softmax statistics
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    max_score = -float('inf')
    sum_exp = 0.0

    # Scale factor for attention
    scale = 1.0 / tl.sqrt(D_HEAD.to(tl.float32))

    # Loop over K, V blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        # Calculate current block bounds
        block_end = tl.minimum(start_n + BLOCK_N, N_CTX)
        block_size = block_end - start_n

        # Create mask for valid elements in this block
        mask = offs_n < block_size

        # Load K, V blocks
        k = tl.load(k_ptrs + start_n * stride_kn, mask=mask[:, None] & (offs_d[None, :] < D_HEAD), other=0.0)
        v = tl.load(v_ptrs + start_n * stride_vn, mask=mask[:, None] & (offs_d[None, :] < D_HEAD), other=0.0)

        # Compute attention scores: q @ K^T (q is 1D, k is 2D)
        scores = tl.sum(q[None, :] * k, axis=1) * scale  # Shape: [BLOCK_N]

        # Apply causal mask if needed (for decoding, usually all positions are valid)
        if IS_CAUSAL:
            causal_mask = start_n + offs_n <= N_CTX - 1  # Current position can attend to all previous
            scores = tl.where(causal_mask & mask, scores, -float('inf'))
        else:
            scores = tl.where(mask, scores, -float('inf'))

        # Online softmax update
        block_max = tl.max(scores, axis=0)

        if block_max > max_score:
            # Rescale previous accumulator
            scale_factor = tl.exp(max_score - block_max)
            acc = acc * scale_factor
            sum_exp = sum_exp * scale_factor
            max_score = block_max

        # Compute probabilities for current block
        probs = tl.exp(scores - max_score)

        # Mask out invalid probabilities
        probs = tl.where(mask, probs, 0.0)

        # Update sum of exponentials
        sum_exp += tl.sum(probs, axis=0)

        # Update accumulator: add weighted values
        acc += tl.sum(probs[:, None] * v, axis=0)

    # Final normalization
    acc = acc / sum_exp

    # Store output
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_d < D_HEAD)

def flash_attention_2(q, k, v, causal=False):
    """
    Flash Attention 2 optimized for decoding (q_seq_len = 1)

    Args:
        q: Query tensor [batch, n_heads, 1, d_head] - single token
        k: Key tensor [batch, n_heads, seq_len_k, d_head] - KV cache
        v: Value tensor [batch, n_heads, seq_len_v, d_head] - KV cache
        causal: Whether to apply causal masking

    Returns:
        Output tensor [batch, n_heads, 1, d_head]
    """
    batch, n_heads, q_seq_len, d_head = q.shape
    kv_seq_len = k.shape[2]

    o = torch.empty_like(q)

    grid = (batch * n_heads,)
    flash_attention_2_decode_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(3),              # Q strides [B, H, 1, D]
        k.stride(0), k.stride(1), k.stride(2), k.stride(3), # K strides [B, H, N, D]
        v.stride(0), v.stride(1), v.stride(2), v.stride(3), # V strides [B, H, N, D]
        o.stride(0), o.stride(1), o.stride(3),              # O strides [B, H, 1, D]
        batch, n_heads, kv_seq_len, d_head,
        BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        IS_CAUSAL=causal
    )

    return o

@triton.jit
def flash_attention_25_decode_kernel(
    Q, K, V, O,
    stride_qb, stride_qh, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_ok,
    B, H, N_CTX, D_HEAD,
    softmax_scale_factor: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # Program IDs
    off_hz = tl.program_id(0)  # (batch * head)
    off_n_block = tl.program_id(1)  # kv block offset

    off_h = off_hz % H
    off_z = off_hz // H
    start_n = off_n_block * BLOCK_N

    # Index ranges
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # Tensor pointers
    q_ptrs = Q + off_z * stride_qb + off_h * stride_qh + offs_d * stride_qk
    k_ptrs = K + off_z * stride_kb + off_h * stride_kh + (start_n + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + off_z * stride_vb + off_h * stride_vh + (start_n + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vk
    o_ptrs = O + off_z * stride_ob + off_h * stride_oh + offs_d * stride_ok

    # Load query
    q = tl.load(q_ptrs, mask=offs_d < D_HEAD, other=0.0)

    # Load key and value blocks
    mask = (start_n + offs_n) < N_CTX
    k = tl.load(k_ptrs, mask=mask[:, None] & (offs_d[None, :] < D_HEAD), other=0.0)
    v = tl.load(v_ptrs, mask=mask[:, None] & (offs_d[None, :] < D_HEAD), other=0.0)

    # Attention scores
    scale = 1.0 / tl.sqrt(D_HEAD.to(tl.float32))
    scores = tl.sum(q[None, :] * k, axis=1) * scale

    # Optional causal masking
    if IS_CAUSAL:
        current_pos = N_CTX - 1  # last token (for decoding)
        causal_mask = (start_n + offs_n) <= current_pos
        scores = tl.where(causal_mask & mask, scores, -float('inf'))
    else:
        scores = tl.where(mask, scores, -float('inf'))

    # Replace max(scores) with provided constant
    scores -= softmax_scale_factor

    # Softmax computation (without sync)
    probs = tl.exp(scores)
    probs = tl.where(mask, probs, 0.0)

    # Output accumulation
    acc = tl.sum(probs[:, None] * v, axis=0)
    sum_exp = tl.sum(probs, axis=0)
    acc = acc / sum_exp

    # Store result
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_d < D_HEAD)

def flash_attention_25(q, k, v, causal=False, softmax_scale_factor=10.0):
    """
    Flash Attention 2.5 for decoding, supporting kv-seq parallelism & fixed softmax shift.

    Args:
        q: [B, H, 1, D]
        k: [B, H, N, D]
        v: [B, H, N, D]
        causal: whether to apply causal mask
        softmax_scale_factor: value to subtract before softmax
        BLOCK_N, BLOCK_D: Triton block sizes

    Returns:
        o: [B, H, 1, D]
    """
    assert q.shape[2] == 1, "This implementation only supports decoding (q_len=1)"
    B, H, _, D = q.shape
    N_CTX = k.shape[2]

    o = torch.empty_like(q)
    grid = (B * H, (N_CTX + BLOCK_N - 1) // BLOCK_N)

    flash_attention_25_decode_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(3),
        B, H, N_CTX, D,
        softmax_scale_factor=softmax_scale_factor,
        BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        IS_CAUSAL=causal,
    )

    return o

def speedometer(fn, *args, warmup, run):
    for _ in range(warmup):
        _ = fn(*args)

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(run):
        output = fn(*args)
    torch.cuda.synchronize()
    end_time = time.time()

    return (end_time - start_time) / run, output

def run_algorithm(algorithm, q, k, v, warmup, run):
    name, fn = algorithm
    try:
        exec_time, output = speedometer(fn, q, k, v, warmup=warmup, run=run)
        return name, exec_time, output
    except Exception as e:
        print(f"{name} failed: {e}")
        return None

def print_results(algorithms, batch_size, n_heads, d_head, kv_seq_len, memory, rtol, atol):
    print(f"\n{'Algorithm':<20} {'Time(ms)':<10} {'Speedup':<10} {'Throughput(M tok/s)':<20} {'Accuracy':<10}")
    print("=" * 80)

    # baseline = Naive Attention
    baseline_time = algorithms[0][1]
    baseline_output = algorithms[0][2]

    for name, exec_time, output in algorithms:
        speedup = f"{(baseline_time / exec_time):.2f}x"
        throughput = f"{((batch_size * kv_seq_len) / exec_time / 1e6):.2f}"
        if torch.allclose(output, baseline_output, rtol=rtol, atol=atol):
            accuracy = "✅ Pass"
        else:
            f"⚠️ {torch.max(torch.abs(output - baseline_output)).item():.1e}"
        print(f"{name:<20} {exec_time * 1000:<10.2f} {speedup:<10} {throughput:<20} {accuracy:<10}")

    print(f"\nConfiguration:")
    print(f"  batch_size={batch_size}, n_heads={n_heads}, d_head={d_head}, kv_seq_len={kv_seq_len}")
    print(f"  KV Cache={memory:.2f} GB")
    print()

def benchmark(configs, algorithms, warmup=5, run=10, rtol=1e-3, atol=1e-3):
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
            q = torch.randn(batch_size, n_heads, 1, d_head, device=device, dtype=dtype)
            k = torch.randn(batch_size, n_heads, kv_seq_len, d_head, device=device, dtype=dtype)
            v = torch.randn(batch_size, n_heads, kv_seq_len, d_head, device=device, dtype=dtype)

            results = []
            for algorithm in algorithms:
                result = run_algorithm(algorithm, q, k, v, warmup, run)
                if result:
                    results.append(result)
            print_results(results, batch_size, n_heads, d_head, kv_seq_len, memory, rtol, atol)

        except Exception as e:
            print(f"Unexpected error in config {config}: {e}")

def main():
    print_device_info()

    #batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    #n_heads = [12, 24, 32, 40, 96]
    #d_heads = [64, 80, 96, 128]
    #kv_seq_lens = [1024, 2048, 4096, 8192, 128000]
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
        #("FA1 (Triton)", flash_attention_1),
        ("FA2 (Triton)", flash_attention_2),
        ("Flash Attention 2", F.scaled_dot_product_attention),
        ("Flash Attention 2.5", flash_attention_25),
        # Add more as needed
    ]
    benchmark(configs=test_configs, algorithms=algorithms)

if __name__ == "__main__":
    main()