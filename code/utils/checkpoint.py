"""
utils package for deepfake detection project.

Contains:
- checkpoint.py: save/load model checkpoints
- metrics.py: compute evaluation metrics
- scheduler.py: learning rate scheduler with warmup
- seed.py: reproducibility utilities
- logger.py: session logging utilities

Each module follows best practices, with docstrings, exception handling,
and edge case safety.
"""

# utils/checkpoint.py
import os
import torch
from typing import Any, Dict


def save_checkpoint(state: Dict[str, Any], is_best: bool, output_dir: str, filename: str = "last.pth") -> str:
    """Save model checkpoint.

    Args:
        state (Dict[str, Any]): State dictionary containing model, optimizer, etc.
        is_best (bool): Whether this is the best checkpoint so far.
        output_dir (str): Directory to save checkpoints.
        filename (str): Name for the last checkpoint file.

    Returns:
        str: Path to the saved checkpoint.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        last_path = os.path.join(output_dir, filename)
        torch.save(state, last_path)
        if is_best:
            best_path = os.path.join(output_dir, "best.pth")
            torch.save(state, best_path)
        return last_path
    except Exception as e:
        raise RuntimeError(f"Failed to save checkpoint: {e}")


def load_checkpoint(filepath: str, map_location: str = "cpu") -> Dict[str, Any]:
    """Load model checkpoint.

    Args:
        filepath (str): Path to the checkpoint.
        map_location (str): Device mapping for torch.load.

    Returns:
        Dict[str, Any]: Loaded checkpoint state.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint not found at {filepath}")
    try:
        return torch.load(filepath, map_location=map_location)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")
