import os
import random
import numpy as np
from typing import Tuple, Optional
import torch
#import transformer_engine.pytorch as te


def set_seed(seed: int):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def load_text(file_path: str, encoding: str = 'utf-8') -> str:
    """
    Load and read text data from a file.

    Args:
        file_path (str): Path to the text file.
        encoding (str, optional): File encoding. Defaults to 'utf-8'.

    Returns:
        str: The content of the text file.
    """
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding=encoding) as f:
        text = f.read()

    print(f"Loaded text data from {file_path} (length: {len(text)} characters).")
    return text

def split_text(text: str, val_size: float) -> Tuple[str, str]:
    """
    Split text into training and validation sets.

    Args:
        text (str): The data to split.
        val_size (float): Size of the validation set.

    Returns:
        Tuple[str, str]: Training and validation data.
    """
    if val_size < 0 or val_size >= 1:
        raise ValueError(f"Invalid validation size: {val_size}")

    split_idx = int(len(text) * (1 - val_size))
    return text[:split_idx], text[split_idx:]


def speedometer(
    module: torch.nn.Module,
    input: torch.Tensor,
    output_grad: torch.Tensor,
    forward_kwargs: dict = {},
    fp8_autocast_kwargs: Optional[dict] = None,
    timing_iters: int = 50,
    warmup_iters: int = 50,
) -> None:
    """Measure average run time for a PyTorch module

    Performs forward and backward passes.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if fp8_autocast_kwargs is None:
        fp8_autocast_kwargs = {"enabled": False}

    # Warmup runs
    torch.cuda.synchronize()
    for _ in range(warmup_iters):
        with te.fp8_autocast(**fp8_autocast_kwargs):
            output = module(input, **forward_kwargs)
        output.backward(output_grad)

    # Timing runs
    start.record()
    for _ in range(timing_iters):
        with te.fp8_autocast(**fp8_autocast_kwargs):
            output = module(input, **forward_kwargs)
        output.backward(output_grad)
    end.record()
    torch.cuda.synchronize()

    print(f"Mean time: {start.elapsed_time(end)/timing_iters} ms")