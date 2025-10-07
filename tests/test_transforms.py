# tests/test_transforms_and_metrics.py
import torch
import numpy as np
from code.transforms import get_train_augment, get_val_transform


def test_transforms_working():
    """Test whether audio transforms produce valid tensors."""
    sample_rate = 16000
    waveform = torch.randn(1, sample_rate * 2)  # 2-second random audio

    train_tf = get_train_augment(sample_rate)
    val_tf = get_val_transform(sample_rate)

    train_out = train_tf(waveform)
    val_out = val_tf(waveform)

    # Check type and shape
    assert isinstance(train_out, torch.Tensor), "Train transform output is not a tensor"
    assert isinstance(val_out, torch.Tensor), "Val transform output is not a tensor"
    assert train_out.dim() == 3, "Expected [channels, n_mels, time] for train transform"
    assert val_out.dim() == 3, "Expected [channels, n_mels, time] for val transform"
    print("✅ Transforms test passed.")




test_transforms_working()
