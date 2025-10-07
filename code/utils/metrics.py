# utils/metrics.py
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from typing import Tuple
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


def compute_eer(y_true, y_score) -> float:
    """Compute Equal Error Rate (EER)."""
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return float(eer)


def compute_metrics(y_true, y_pred, y_prob) -> Tuple[float, float, float, float, float, float]:
    """Compute classification metrics for audio deepfake detection.

    Args:
        y_true (list or np.ndarray): True labels (0=bonafide, 1=spoof).
        y_pred (list or np.ndarray): Predicted labels.
        y_prob (list or np.ndarray): Predicted probabilities for spoof class.

    Returns:
        Tuple: (accuracy, auc, precision, recall, f1, eer)
    """
    try:
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        eer = compute_eer(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
        return acc, auc, precision, recall, f1, eer

    except Exception as e:
        raise RuntimeError(f"Failed to compute metrics: {e}")