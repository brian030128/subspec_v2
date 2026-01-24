"""Shared utilities for benchmark pipelines."""

import os
import gc
import random
import torch

from run.core.config_utils import write_settings_yaml


def reset_seeds(seed: int = 0) -> None:
    """Reset random seeds for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)


def cleanup_gpu() -> None:
    """Clean up GPU memory."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()


def setup_benchmark_dir(log_dir_base: str, bench_name: str, settings_snapshot=None) -> str:
    """Create benchmark output directory and write settings.
    
    Args:
        log_dir_base: Base directory for logs
        bench_name: Name of the benchmark
        settings_snapshot: Optional settings dict to write
        
    Returns:
        Path to the created benchmark directory
    """
    log_dir = os.path.join(log_dir_base, bench_name)
    os.makedirs(log_dir, exist_ok=True)
    write_settings_yaml(log_dir, settings_snapshot)
    return log_dir
