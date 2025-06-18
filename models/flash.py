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
        ("FA1 (Triton)", flash_attention_1),
        ("FA2 (Triton)", flash_attention_2),
        ("Flash Attention 2", F.scaled_dot_product_attention),
        ("Flash Attention 2.5", F.scaled_dot_product_attention),
        # Add more as needed
    ]
    benchmark(configs=test_configs, algorithms=algorithms)

if __name__ == "__main__":
    main()