# code/transforms.py
"""
Transforms module for audio deepfake detection.

Provides training and validation transform pipelines using torchaudio.
Includes common augmentations like noise addition, time masking, and frequency masking.
"""

from typing import Callable
import torch
import torchaudio
import random


class AddNoise:
    """Add random Gaussian noise to waveform."""
    def __init__(self, noise_level: float = 0.005):
        self.noise_level = noise_level

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(waveform) * self.noise_level
        return waveform + noise


class RandomGain:
    """Randomly amplify or attenuate the audio signal."""
    def __init__(self, min_gain: float = 0.8, max_gain: float = 1.2):
        self.min_gain = min_gain
        self.max_gain = max_gain

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        gain = random.uniform(self.min_gain, self.max_gain)
        return waveform * gain


def get_train_transform(sample_rate: int = 16000) -> Callable:
    """Return transform pipeline for training audio data."""
    return torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=64),
        torchaudio.transforms.AmplitudeToDB(),
    )


def get_train_augment(sample_rate: int = 16000) -> Callable:
    """Return a transform pipeline with augmentations for training."""
    def augment(waveform: torch.Tensor) -> torch.Tensor:
        # Random augmentations on waveform
        if random.random() < 0.5:
            waveform = AddNoise()(waveform)
        if random.random() < 0.5:
            waveform = RandomGain()(waveform)

        # Convert to log-mel spectrogram
        mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=64)(waveform)
        mel_db = torchaudio.transforms.AmplitudeToDB()(mel)

        # Apply time and frequency masking
        mel_db = torchaudio.transforms.FrequencyMasking(freq_mask_param=8)(mel_db)
        mel_db = torchaudio.transforms.TimeMasking(time_mask_param=10)(mel_db)

        return mel_db

    return augment


def get_val_transform(sample_rate: int = 16000) -> Callable:
    """Return transform pipeline for validation/testing (no augmentation)."""
    def transform(waveform: torch.Tensor) -> torch.Tensor:
        mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=64)(waveform)
        mel_db = torchaudio.transforms.AmplitudeToDB()(mel)
        return mel_db

    return transform
