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

# utils/metrics.py
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
import numpy as np
from typing import Tuple


def compute_metrics(y_true, y_pred, y_prob) -> Tuple[float, float, float, float, float]:
    """Compute classification metrics.

    Args:
        y_true (list or np.ndarray): True labels.
        y_pred (list or np.ndarray): Predicted labels.
        y_prob (list or np.ndarray): Predicted probabilities for positive class.

    Returns:
        Tuple: (accuracy, auc, precision, recall, f1)
    """
    try:
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        return acc, auc, precision, recall, f1
    except Exception as e:
        raise RuntimeError(f"Failed to compute metrics: {e}")


