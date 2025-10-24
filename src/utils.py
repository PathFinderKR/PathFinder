import os
import random
import numpy as np
from typing import Tuple
import torch
import torch.nn as nn


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
    model: nn.Module,
    input_ids: torch.Tensor,
    use_cache: bool,
    warmup_tokens: int = 1000,
    timing_tokens: int = 1000,
    num_runs: int = 10,
    **generate_kwargs
) -> None:
    """
    Measure inference speed of a model.

    Args:
        model (nn.Module): The model to measure speed for.
        input_ids (torch.Tensor): Input IDs for the model.
        use_cache (bool): Whether to use cache for generation.
        warmup_tokens (int): Number of tokens to warmup the model. Defaults to 100.
        timing_tokens (int): Number of tokens to time the model. Defaults to 100.
        num_runs (int): Number of runs to measure speed. Defaults to 5.
        **generate_kwargs: Additional keyword arguments for model generation.
    """
    print(f"KV Cache Enabled: {use_cache}")
    print(f"Warmup Tokens: {warmup_tokens}, Timing Tokens: {timing_tokens}, Runs: {num_runs}")
    print("-" * 50)

    # Warmup runs
    model.generate(
        input_ids,
        use_cache=use_cache,
        max_new_tokens=warmup_tokens,
        **generate_kwargs
    )

    # Multiple timing runs
    times = []
    for i in range(num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start.record()
        model.generate(
            input_ids,
            use_cache=use_cache,
            max_new_tokens=timing_tokens,
            **generate_kwargs
        )
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        times.append(elapsed_ms)

        latency = elapsed_ms / timing_tokens
        throughput = timing_tokens / (elapsed_ms / 1000)

        print(f"Run {i + 1:2d}: Latency = {latency:.2f} ms/token, Throughput = {throughput:.2f} tokens/sec")

    print("-" * 50)
    times_np = np.array(times)
    avg_time = times_np.mean()
    std_dev = times_np.std()
    min_time = times_np.min()
    max_time = times_np.max()
    median_time = np.median(times_np)

    print(f"Summary (over {num_runs} runs):")
    print(f"  Avg    Latency: {avg_time / timing_tokens:.2f} ms/token")
    print(f"  Std    Latency: {std_dev / timing_tokens:.2f} ms/token")
    print(f"  Min    Latency: {min_time / timing_tokens:.2f} ms/token")
    print(f"  Max    Latency: {max_time / timing_tokens:.2f} ms/token")
    print(f"  Median Latency: {median_time / timing_tokens:.2f} ms/token")
    print(f"  Avg    Throughput: {timing_tokens / (avg_time / 1000):.2f} tokens/sec")