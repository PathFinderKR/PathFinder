import time
import math
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

BLOCK_N = 128
NUM_WARPS = 4
NUM_STAGES = 2

@triton.jit
def _flash_attn_decode_kernel(
    Q,     # [B, H, Tq=1, D]
    K, V,  # [B, H, T,    D]
    O,     # [B, H, To=1, D]
    scale,
    stride_Q_B, stride_Q_H, stride_Q_T, stride_Q_D,
    stride_K_B, stride_K_H, stride_K_T, stride_K_D,
    stride_V_B, stride_V_H, stride_V_T, stride_V_D,
    stride_O_B, stride_O_H, stride_O_T, stride_O_D,
    T,                     # kv_seq_len
    D: tl.constexpr,       # d_head
    BLOCK_N: tl.constexpr
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)

    offs_d = tl.arange(0, D)

    # Q[bid, hid, 0, :]
    q_ptr = Q + bid * stride_Q_B + hid * stride_Q_H + 0 * stride_Q_T + offs_d * stride_Q_D
    q = tl.load(q_ptr)                       # [D], dtype(Q)
    q_f32 = q.to(tl.float32)

    running_max = tl.full([1], -float("inf"), dtype=tl.float32)
    running_sum = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([D], dtype=tl.float32)

    for start_n in range(0, T, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_rows = (offs_n[:, None] < T)

        # K[bid, hid, offs_n, :]
        k_ptrs = (K
                  + bid * stride_K_B
                  + hid * stride_K_H
                  + offs_n[:, None] * stride_K_T
                  + offs_d[None, :] * stride_K_D)
        k = tl.load(k_ptrs, mask=mask_rows, other=0.0)        # [BLOCK_N, D]
        k_f32 = k.to(tl.float32)

        # qk = dot(q, k_i)  (tl.dot은 2D/3D만 허용 → sum로 계산)
        qk = tl.sum(k_f32 * q_f32[None, :], axis=1) * scale   # [BLOCK_N]
        qk = tl.where(offs_n < T, qk, float("-inf"))

        # V[bid, hid, offs_n, :]
        v_ptrs = (V
                  + bid * stride_V_B
                  + hid * stride_V_H
                  + offs_n[:, None] * stride_V_T
                  + offs_d[None, :] * stride_V_D)
        v = tl.load(v_ptrs, mask=mask_rows, other=0.0)        # [BLOCK_N, D]
        v_f32 = v.to(tl.float32)

        # Online softmax
        tile_max = tl.max(qk, axis=0)
        new_running_max = tl.maximum(tile_max, running_max)

        old_scale = tl.exp(running_max - new_running_max)
        acc = acc * old_scale
        running_sum = running_sum * old_scale

        p = tl.exp(qk - new_running_max)                      # [BLOCK_N]
        acc += tl.sum(p[:, None] * v_f32, axis=0)             # [D]
        running_sum += tl.sum(p, axis=0)                      # [1]

        running_max = new_running_max

    eps = 1e-6
    o = acc / (running_sum + eps)                             # [D], fp32
    o_ptr = O + bid * stride_O_B + hid * stride_O_H + 0 * stride_O_T + offs_d * stride_O_D
    tl.store(o_ptr, o)  # dtype(O)로 자동 캐스트


def flash_attn_decode(
    q,             # [B, H, 1, D]
    k, v,          # [B, H, T, D]
    scale: float
):
    B, H, Tkv, D = k.size()
    o = torch.empty_like(q)

    stride_q_b, stride_q_h, stride_q_t, stride_q_d = q.stride()
    stride_k_b, stride_k_h, stride_k_t, stride_k_d = k.stride()
    stride_v_b, stride_v_h, stride_v_t, stride_v_d = v.stride()
    stride_o_b, stride_o_h, stride_o_t, stride_o_d = o.stride()

    grid = (B, H)
    _flash_attn_decode_kernel[grid](
        q, k, v, o,
        scale,
        stride_q_b, stride_q_h, stride_q_t, stride_q_d,
        stride_k_b, stride_k_h, stride_k_t, stride_k_d,
        stride_v_b, stride_v_h, stride_v_t, stride_v_d,
        stride_o_b, stride_o_h, stride_o_t, stride_o_d,
        Tkv,
        D=D,
        BLOCK_N=BLOCK_N,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES
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