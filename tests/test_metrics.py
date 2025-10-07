# tests/test_transforms_and_metrics.py
import torch
import numpy as np
from code.utils.metrics import compute_metrics


def test_metrics_working():
    """Test whether metrics can be computed without errors."""
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.3, 0.8, 0.7, 0.2])
    y_pred = (y_prob > 0.5).astype(int)

    results = compute_metrics(y_true, y_pred, y_prob)
    acc, auc, precision, recall, f1, eer = results

    assert 0.0 <= acc <= 1.0, "Invalid accuracy value"
    assert 0.0 <= auc <= 1.0 or np.isnan(auc), "Invalid AUC value"
    assert 0.0 <= precision <= 1.0, "Invalid precision"
    assert 0.0 <= recall <= 1.0, "Invalid recall"
    assert 0.0 <= f1 <= 1.0, "Invalid F1"
    assert 0.0 <= eer <= 1.0 or np.isnan(eer), "Invalid EER"

    print("✅ Metrics test passed.")


test_metrics_working()