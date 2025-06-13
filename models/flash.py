import torch
import triton
import triton.language as tl
from triton.runtime import driver

device = torch.device("cuda")
DEVICE = driver.active.get_current_device()
properties = driver.active.utils.get_device_properties(DEVICE)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()




def main():
    print(f"Device: {torch.cuda.get_device_name(device)}\n"
          f"Number of SM: {NUM_SM}\n"
          f"Number of registers: {NUM_REGS}\n"
          f"Size of SMEM: {SIZE_SMEM}\n"
          f"Warp size: {WARP_SIZE}\n"
          f"Target: {target}")


if __name__ == "__main__":
    main()