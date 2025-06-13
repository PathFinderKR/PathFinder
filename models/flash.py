import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import math
import time


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 16, 'BLOCK_D': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 32, 'BLOCK_D': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_D': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_D': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_D': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 32, 'BLOCK_D': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_D': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_D': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_D': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_D': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_D': 256}, num_warps=8, num_stages=2),
    ],
    key=['N', 'D'],  # 시퀀스 길이와 d_head에 따라 자동 최적화
)
@triton.jit
def autotuned_flash_decode_kernel(
        Q, K, V, Out,
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_od,
        B, H, N, D,
        scale,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
):
    """
    Triton Autotuned Flash Decode Kernel
    - 자동으로 최적 블록 크기 선택
    - 다양한 하드웨어에서 최적 성능
    """
    # 작업 ID
    pid = tl.program_id(0)
    batch_id = pid // H
    head_id = pid % H

    # Q 벡터 로드
    q_offset = batch_id * stride_qb + head_id * stride_qh
    d_range = tl.arange(0, BLOCK_D)
    q = tl.load(Q + q_offset + d_range, mask=d_range < D)

    # 누적 변수
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    max_score = -float('inf')
    sum_exp = 0.0

    # KV 블록 처리
    for start_n in tl.range(0, N, BLOCK_N):
        end_n = tl.minimum(start_n + BLOCK_N, N)

        # K 블록 로드
        k_offset = batch_id * stride_kb + head_id * stride_kh + start_n * stride_kn
        n_range = tl.arange(0, BLOCK_N)
        k_ptrs = K + k_offset + n_range[:, None] * stride_kn + d_range[None, :]
        k_mask = (n_range[:, None] < (end_n - start_n)) & (d_range[None, :] < D)
        k_vals = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Attention scores
        scores = tl.sum(q[None, :] * k_vals, axis=1) * scale
        scores = tl.where(n_range < (end_n - start_n), scores, -float('inf'))

        # Online softmax
        block_max = tl.max(scores)
        new_max = tl.maximum(max_score, block_max)

        if max_score > -float('inf'):
            exp_diff = tl.exp(max_score - new_max)
            acc = acc * exp_diff
            sum_exp = sum_exp * exp_diff

        # Softmax weights
        weights = tl.exp(scores - new_max)
        weights = tl.where(n_range < (end_n - start_n), weights, 0.0)
        block_sum = tl.sum(weights)

        # V 블록 로드 및 누적
        v_offset = batch_id * stride_vb + head_id * stride_vh + start_n * stride_vn
        v_ptrs = V + v_offset + n_range[:, None] * stride_vn + d_range[None, :]
        v_mask = (n_range[:, None] < (end_n - start_n)) & (d_range[None, :] < D)
        v_vals = tl.load(v_ptrs, mask=v_mask, other=0.0)

        weighted_v = tl.sum(weights[:, None] * v_vals, axis=0)
        acc = acc + weighted_v
        sum_exp = sum_exp + block_sum
        max_score = new_max

    # 출력
    result = acc / tl.maximum(sum_exp, 1e-8)
    out_offset = batch_id * stride_ob + head_id * stride_oh
    tl.store(Out + out_offset + d_range, result, mask=d_range < D)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 32, 'BLOCK_D': 64, 'BLOCK_BH': 1}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_D': 64, 'BLOCK_BH': 1}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_D': 64, 'BLOCK_BH': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_D': 64, 'BLOCK_BH': 2}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_D': 64, 'BLOCK_BH': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_D': 128, 'BLOCK_BH': 1}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_D': 128, 'BLOCK_BH': 2}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_D': 128, 'BLOCK_BH': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_D': 256, 'BLOCK_BH': 2}, num_warps=8, num_stages=2),
    ],
    key=['B', 'H', 'N', 'D'],  # 모든 차원을 고려한 최적화
)
@triton.jit
def mega_autotuned_flash_kernel(
        Q, K, V, Out,
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_od,
        B, H, N, D,
        scale,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_BH: tl.constexpr,
):
    """
    Mega Autotuned Flash Kernel
    - 배치/헤드 블로킹도 자동 최적화
    - 대규모 워크로드 특화
    """
    pid = tl.program_id(0)

    # 블록당 여러 배치×헤드 처리
    for local_idx in range(BLOCK_BH):
        bh_id = pid * BLOCK_BH + local_idx
        if bh_id >= B * H:
            break

        batch_id = bh_id // H
        head_id = bh_id % H

        # Q 로드
        q_offset = batch_id * stride_qb + head_id * stride_qh
        d_range = tl.arange(0, BLOCK_D)
        q = tl.load(Q + q_offset + d_range, mask=d_range < D)

        # Flash attention 계산
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        max_score = -float('inf')
        sum_exp = 0.0

        for start_n in tl.range(0, N, BLOCK_N):
            end_n = tl.minimum(start_n + BLOCK_N, N)
            block_size = end_n - start_n

            # K 로드
            k_offset = batch_id * stride_kb + head_id * stride_kh + start_n * stride_kn
            n_range = tl.arange(0, BLOCK_N)
            k_ptrs = K + k_offset + n_range[:, None] * stride_kn + d_range[None, :]
            k_mask = (n_range[:, None] < block_size) & (d_range[None, :] < D)
            k_vals = tl.load(k_ptrs, mask=k_mask, other=0.0)

            # Scores
            scores = tl.sum(q[None, :] * k_vals, axis=1) * scale
            scores = tl.where(n_range < block_size, scores, -float('inf'))

            # Softmax
            block_max = tl.max(scores)
            new_max = tl.maximum(max_score, block_max)

            if max_score > -float('inf'):
                scale_factor = tl.exp(max_score - new_max)
                acc = acc * scale_factor
                sum_exp = sum_exp * scale_factor

            weights = tl.exp(scores - new_max)
            weights = tl.where(n_range < block_size, weights, 0.0)
            block_sum = tl.sum(weights)

            # V 로드 및 누적
            v_offset = batch_id * stride_vb + head_id * stride_vh + start_n * stride_vn
            v_ptrs = V + v_offset + n_range[:, None] * stride_vn + d_range[None, :]
            v_mask = (n_range[:, None] < block_size) & (d_range[None, :] < D)
            v_vals = tl.load(v_ptrs, mask=v_mask, other=0.0)

            weighted_v = tl.sum(weights[:, None] * v_vals, axis=0)
            acc = acc + weighted_v
            sum_exp = sum_exp + block_sum
            max_score = new_max

        # 출력 저장
        result = acc / tl.maximum(sum_exp, 1e-8)
        out_offset = batch_id * stride_ob + head_id * stride_oh
        tl.store(Out + out_offset + d_range, result, mask=d_range < D)


def autotuned_flash_attn_decode(q, k, v, scale=None):
    """Autotuned Flash Attention Decode"""
    batch_size, n_heads, q_seq_len, d_head = q.shape
    _, _, kv_seq_len, _ = k.shape

    assert q_seq_len == 1, "Decode mode requires q_seq_len=1"

    if scale is None:
        scale = 1.0 / math.sqrt(d_head)

    out = torch.empty_like(q)
    grid = (batch_size * n_heads,)

    autotuned_flash_decode_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(3),
        batch_size, n_heads, kv_seq_len, d_head,
        scale,
    )

    return out


def mega_autotuned_flash_attn_decode(q, k, v, scale=None):
    """Mega Autotuned Flash Attention - 대규모 배치 특화"""
    batch_size, n_heads, q_seq_len, d_head = q.shape
    _, _, kv_seq_len, _ = k.shape

    if scale is None:
        scale = 1.0 / math.sqrt(d_head)

    out = torch.empty_like(q)

    # 자동 그리드 크기 계산
    total_bh = batch_size * n_heads
    # Autotune이 BLOCK_BH를 결정하므로 충분히 큰 그리드 설정
    grid_size = min(total_bh, 1024)  # 최대 1024 블록

    mega_autotuned_flash_kernel[(grid_size,)](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(3),
        batch_size, n_heads, kv_seq_len, d_head,
        scale,
    )

    return out


def naive_attention(q, k, v, scale=None):
    """PyTorch 기본 구현"""
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = F.softmax(scores, dim=-1)
    out = torch.matmul(attn_weights, v)
    return out


def comprehensive_scale_benchmark():
    """포괄적 스케일 벤치마크 - batch_size=1 포함"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not torch.cuda.is_available():
        print("❌ CUDA 필요")
        return

    # 다양한 스케일 테스트 (batch_size=1 포함)
    test_configs = [
        # Single batch 시나리오 (추론 서버)
        {"name": "Single Batch - Small", "batch_size": 1, "n_heads": 12, "kv_seq_len": 2048, "d_head": 64},
        {"name": "Single Batch - Medium", "batch_size": 1, "n_heads": 32, "kv_seq_len": 8192, "d_head": 128},
        {"name": "Single Batch - Large", "batch_size": 1, "n_heads": 40, "kv_seq_len": 16384, "d_head": 128},
        {"name": "Single Batch - XL", "batch_size": 1, "n_heads": 64, "kv_seq_len": 32768, "d_head": 128},

        # Small batch 시나리오
        {"name": "Small Batch - GPT-3", "batch_size": 4, "n_heads": 96, "kv_seq_len": 4096, "d_head": 128},
        {"name": "Small Batch - GPT-4", "batch_size": 8, "n_heads": 128, "kv_seq_len": 8192, "d_head": 128},

        # Medium batch 시나리오
        {"name": "Medium Batch - Service", "batch_size": 32, "n_heads": 64, "kv_seq_len": 4096, "d_head": 128},
        {"name": "Medium Batch - Training", "batch_size": 64, "n_heads": 80, "kv_seq_len": 8192, "d_head": 128},

        # Large batch 시나리오
        {"name": "Large Batch - Datacenter", "batch_size": 128, "n_heads": 96, "kv_seq_len": 4096, "d_head": 128},
        {"name": "Large Batch - Extreme", "batch_size": 256, "n_heads": 64, "kv_seq_len": 2048, "d_head": 128},

        # 초장문 컨텍스트
        {"name": "Ultra Long Context", "batch_size": 2, "n_heads": 32, "kv_seq_len": 131072, "d_head": 128},
        # 128K tokens
    ]

    print("🚀 포괄적 Autotuned Flash Attention 벤치마크")
    print("=" * 100)

    for config in test_configs:
        name = config["name"]
        batch_size = config["batch_size"]
        n_heads = config["n_heads"]
        kv_seq_len = config["kv_seq_len"]
        d_head = config["d_head"]

        # 메모리 계산
        kv_memory_gb = (batch_size * n_heads * kv_seq_len * d_head * 2 * 4) / (1024 ** 3)

        print(f"\n🎯 {name}")
        print(f"📊 B={batch_size}, H={n_heads}, KV_len={kv_seq_len:,}, D={d_head}")
        print(f"💾 KV Cache: {kv_memory_gb:.2f} GB")

        # 메모리 제한 체크
        if kv_memory_gb > 24:  # 24GB 제한
            print("⚠️ 메모리 제한으로 스킵")
            continue

        try:
            # 데이터 준비
            dtype = torch.float16 if kv_memory_gb > 4 else torch.float32
            q = torch.randn(batch_size, n_heads, 1, d_head, device=device, dtype=dtype)
            k = torch.randn(batch_size, n_heads, kv_seq_len, d_head, device=device, dtype=dtype)
            v = torch.randn(batch_size, n_heads, kv_seq_len, d_head, device=device, dtype=dtype)

            scale = 1.0 / math.sqrt(d_head)

            methods = []
            warmup = 3
            runs = 10

            # 1. Naive PyTorch
            try:
                for _ in range(warmup):
                    _ = naive_attention(q, k, v, scale)
                torch.cuda.synchronize()

                start = time.time()
                for _ in range(runs):
                    out_naive = naive_attention(q, k, v, scale)
                torch.cuda.synchronize()

                naive_time = (time.time() - start) / runs
                methods.append(("Naive PyTorch", naive_time, out_naive))
            except Exception as e:
                print(f"❌ Naive 실패: {e}")

            # 2. PyTorch Flash Attention 2
            try:
                for _ in range(warmup):
                    _ = F.scaled_dot_product_attention(q, k, v, scale=scale)
                torch.cuda.synchronize()

                start = time.time()
                for _ in range(runs):
                    out_fa2 = F.scaled_dot_product_attention(q, k, v, scale=scale)
                torch.cuda.synchronize()

                fa2_time = (time.time() - start) / runs
                methods.append(("PyTorch FA2", fa2_time, out_fa2))
            except Exception as e:
                print(f"❌ PyTorch FA2 실패: {e}")

            # 3. Autotuned Flash Decode
            try:
                print("🔧 Triton autotune 진행 중...")
                for _ in range(warmup):
                    _ = autotuned_flash_attn_decode(q, k, v, scale)
                torch.cuda.synchronize()

                start = time.time()
                for _ in range(runs):
                    out_auto = autotuned_flash_attn_decode(q, k, v, scale)
                torch.cuda.synchronize()

                auto_time = (time.time() - start) / runs
                methods.append(("Autotuned Triton", auto_time, out_auto))
            except Exception as e:
                print(f"❌ Autotuned 실패: {e}")

            # 4. Mega Autotuned (배치 크기 >= 4만)
            if batch_size >= 4:
                try:
                    print("🔧 Mega autotune 진행 중...")
                    for _ in range(warmup):
                        _ = mega_autotuned_flash_attn_decode(q, k, v, scale)
                    torch.cuda.synchronize()

                    start = time.time()
                    for _ in range(runs):
                        out_mega = mega_autotuned_flash_attn_decode(q, k, v, scale)
                    torch.cuda.synchronize()

                    mega_time = (time.time() - start) / runs
                    methods.append(("Mega Autotuned", mega_time, out_mega))
                except Exception as e:
                    print(f"❌ Mega Autotuned 실패: {e}")

            # 결과 출력
            if methods:
                print(f"\n{'구현':<18} {'시간(ms)':<10} {'속도향상':<10} {'처리량(M tok/s)':<15} {'정확도'}")
                print("-" * 78)

                baseline_time = methods[0][1]
                baseline_output = methods[0][2]

                for name, exec_time, output in methods:
                    speedup = f"{baseline_time / exec_time:.1f}x"

                    # 토큰 처리량 (M tokens/sec)
                    total_tokens = batch_size * kv_seq_len
                    throughput_m = (total_tokens / exec_time) / 1e6

                    # 정확도 체크
                    try:
                        if torch.allclose(output, baseline_output, rtol=1e-2, atol=1e-2):
                            accuracy = "✅"
                        else:
                            diff = torch.max(torch.abs(output - baseline_output)).item()
                            accuracy = f"⚠️{diff:.1e}"
                    except:
                        accuracy = "❓"

                    print(f"{name:<18} {exec_time * 1000:<9.1f} {speedup:<10} {throughput_m:<14.1f} {accuracy}")

        except torch.cuda.OutOfMemoryError:
            print("❌ GPU 메모리 부족")
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")


if __name__ == "__main__":
    comprehensive_scale_benchmark()