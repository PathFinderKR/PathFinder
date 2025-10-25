import time
import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.runtime import driver
from src.utils import set_seed
MEMORY_LIMIT_GB: int = 16
BLOCK_T = 128
BLOCK_D = 64
NUM_WARPS = 8
NUM_STAGES = 2

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
    off_h = off_hb % H
    off_b = off_hb // H

    # Offsets
    offs_t = tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)

    # Pointers
    q_ptrs = Q + off_b * stride_q_b + off_h * stride_q_h + 0 * stride_q_t + offs_d * stride_q_d
    k_base = K + off_b * stride_k_b + off_h * stride_k_h
    v_base = V + off_b * stride_v_b + off_h * stride_v_h
    o_ptrs = O + off_b * stride_o_b + off_h * stride_o_h + 0 * stride_o_t + offs_d * stride_o_d

    # Load query vector
    q = tl.load(q_ptrs, mask=offs_d < D, other=0.0)  # [BLOCK_D] (bf16)
    q_fp32 = q.to(tl.float32)

    # accumulator, online-softmax stats (fp32)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    max_score = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    # Scale (fp32)
    Df = tl.full((), 0.0, dtype=tl.float32) + D
    scale = 1.0 / tl.sqrt(Df)
    neg_inf = tl.full((), -float("inf"), dtype=tl.float32)

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
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)  # [BLOCK_T, BLOCK_D]
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0)  # [BLOCK_T, BLOCK_D]
        k_fp32 = k.to(tl.float32)
        v_fp32 = v.to(tl.float32)

        # Compute attention scores: q @ K^T
        scores = tl.sum(k_fp32 * q_fp32[None, :], axis=1) * scale  # [BLOCK_T] (fp32)
        scores = tl.where(mask_t, scores, neg_inf)

        # Online-softmax update (fp32)
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(max_score, block_max)
        rescale = tl.where(max_score == neg_inf, 0.0, tl.exp(max_score - new_max))
        acc *= rescale
        sum_exp *= rescale

        # Probabilities
        probs = tl.exp(scores - new_max)
        probs = tl.where(mask_t, probs, 0.0)

        # Update accumulator and sum
        acc += tl.sum(probs[:, None] * v_fp32, axis=0)  # [BLOCK_D]
        sum_exp += tl.sum(probs, axis=0)
        max_score = new_max

    # Final normalization
    denom = tl.maximum(sum_exp, 1e-20)
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
    FIXMAX, # scalar (fp32)
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
    off_h = off_hb % H
    off_b = off_hb // H

    # Offsets
    offs_t = tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)

    # Pointers
    q_ptrs = Q + off_b * stride_q_b + off_h * stride_q_h + 0 * stride_q_t + offs_d * stride_q_d
    k_base = K + off_b * stride_k_b + off_h * stride_k_h
    v_base = V + off_b * stride_v_b + off_h * stride_v_h
    o_ptrs = O + off_b * stride_o_b + off_h * stride_o_h + 0 * stride_o_t + offs_d * stride_o_d

    # Load query vector
    q = tl.load(q_ptrs, mask=offs_d < D, other=0.0)  # [BLOCK_D] (bf16)
    q_fp32 = q.to(tl.float32)

    # accumulator, online-softmax stats (fp32)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    # Scale (fp32)
    Df = tl.full((), 0.0, dtype=tl.float32) + D
    scale = 1.0 / tl.sqrt(Df)
    neg_inf = tl.full((), -float("inf"), dtype=tl.float32)

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
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)  # [BLOCK_T, BLOCK_D]
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0)  # [BLOCK_T, BLOCK_D]
        k_fp32 = k.to(tl.float32)
        v_fp32 = v.to(tl.float32)

        # Compute attention scores: q @ K^T
        scores = tl.sum(k_fp32 * q_fp32[None, :], axis=1) * scale  # [BLOCK_T] (fp32)
        scores = tl.where(mask_t, scores, neg_inf)

        # Fixed-max softmax probs
        probs = tl.exp(scores - fixmax)  # [BLOCK_T]
        probs = tl.where(mask_t, probs, 0.0)

        # Update accumulator and sum
        acc += tl.sum(probs[:, None] * v_fp32, axis=0)  # [BLOCK_D]
        sum_exp += tl.sum(probs, axis=0)

    # Final normalization
    denom = tl.maximum(sum_exp, 1e-20)
    out = acc / denom

    # Store output
    tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=offs_d < D)

def flash_attn_2_fixmax(q, k, v, fixmax: float = 10):
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
        fixmax,
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
    dtype: torch.dtype, device: torch.device,
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
        if memory > MEMORY_LIMIT_GB:
            print("\033[95m" + "═" * 75 + "\033[0m")
            print("\033[95m🧩 Configuration\033[0m")
            print(f"  • Batch Size : {batch_size}")
            print(f"  • Num Heads  : {n_heads}")
            print(f"  • Head Dim   : {d_head}")
            print(f"  • KV Seq Len : {kv_seq_len}")
            print(f"  • KV Cache   : {memory:.2f} GB")
            print(f"  Memory limit exceeded: {memory:.2f} GB > {MEMORY_LIMIT_GB:.2f} GB\n")
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
    torch.set_float32_matmul_precision("highest")
    dtype = torch.float32
    properties = driver.active.utils.get_device_properties(driver.active.get_current_device())
    num_sm = properties["multiprocessor_count"]
    num_regs = properties["max_num_regs"]
    sram_size = properties["max_shared_mem"]
    warp_size = properties["warpSize"]
    print(f"Device: {torch.cuda.get_device_name(device)}\n"
          f"Number of SM: {num_sm}\n"
          f"Number of registers: {num_regs}\n"
          f"Size of SMEM: {sram_size}\n"
          f"Warp size: {warp_size}")
    set_seed(42)

    # batch_sizes = [1, 4, 32]
    # n_heads     = [12, 16, 16, 24]
    # d_heads     = [64, 64, 96, 128]
    # kv_seq_lens = [128, 256, 1024, 8192]
    test_configs = [
        # Single batch
        {"batch_size": 1, "n_heads": 12, "d_head": 64, "kv_seq_len": 128},
        {"batch_size": 1, "n_heads": 16, "d_head": 64, "kv_seq_len": 256},
        {"batch_size": 1, "n_heads": 16, "d_head": 96, "kv_seq_len": 1024},
        {"batch_size": 1, "n_heads": 24, "d_head": 128, "kv_seq_len": 8192},
        # Small batch
        {"batch_size": 4, "n_heads": 12, "d_head": 64, "kv_seq_len": 128},
        {"batch_size": 4, "n_heads": 16, "d_head": 64, "kv_seq_len": 256},
        {"batch_size": 4, "n_heads": 16, "d_head": 96, "kv_seq_len": 1024},
        {"batch_size": 4, "n_heads": 24, "d_head": 128, "kv_seq_len": 8192},
        # Large batch
        {"batch_size": 32, "n_heads": 12, "d_head": 64, "kv_seq_len": 128},
        {"batch_size": 32, "n_heads": 16, "d_head": 64, "kv_seq_len": 258},
        {"batch_size": 32, "n_heads": 16, "d_head": 86, "kv_seq_len": 1024},
        {"batch_size": 32, "n_heads": 24, "d_head": 128, "kv_seq_len": 8192},
        # Long Context
        {"batch_size": 1, "n_heads": 12, "d_head": 64, "kv_seq_len": 100000},
        {"batch_size": 2, "n_heads": 12, "d_head": 64, "kv_seq_len": 100000}
    ]
    algorithms = [
        ("Naive Attention", naive_attention),
        # ("Flash Attention (Triton)", flash_attn_1),
        ("FA2", F.scaled_dot_product_attention),
        ("FA2 (Triton)", flash_attn_2),
        ("FA2 + fix-max", flash_attn_2_fixmax),
        #("Flash Decoding", flash_decoding),
        #("Flash Decoding 2",flash_decoding_2),
    ]
    benchmark(
        configs=test_configs, algorithms=algorithms,
        dtype=dtype, device=device
    )


if __name__ == "__main__":
    main()