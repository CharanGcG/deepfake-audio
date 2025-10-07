from typing import Optional, Callable, Tuple, Dict, Any
from torch.utils.data import Dataset
import pandas as pd
import os
import torch
import torchaudio


class AudioDataset(Dataset):
    def __init__(self, csv_path: str, root_dir: str, transform: Optional[Callable] = None,
                 target_sample_rate: int = 16000, max_length: Optional[float] = None, logger=None):
        """
        Dataset for loading audio files (.flac) for deepfake detection.

        Args:
            csv_path: Path to CSV file (must contain 'path' and 'label' columns)
            root_dir: Base directory for audio file paths in CSV
            transform: Optional callable to apply on waveform tensor (e.g., spectrogram)
            target_sample_rate: Resample all audio to this rate (Hz)
            max_length: Pad/crop audio to this length in seconds (None disables)
        """
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        if "path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("CSV must contain 'path' and 'label' columns")

        self.root_dir = root_dir
        self.transform = transform
        self.sample_rate = target_sample_rate
        self.max_length = max_length

        self.df = self.df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _load_audio(self, rel_path: str) -> torch.Tensor:
        normalized_path = os.path.normpath(rel_path)
        full_path = os.path.join(self.root_dir, normalized_path)

        if not os.path.isfile(full_path):
            print(f"Warning: audio file not found: {full_path}")
            # Return silent audio (all zeros) of max_length or 1 second default
            duration = self.max_length or 1.0
            num_samples = int(duration * self.sample_rate)
            return torch.zeros(1, num_samples)

        try:
            waveform, sr = torchaudio.load(full_path)
            waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono

            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)

            # Crop or pad waveform to max_length if specified
            if self.max_length is not None:
                max_len_samples = int(self.sample_rate * self.max_length)
                if waveform.shape[1] > max_len_samples:
                    waveform = waveform[:, :max_len_samples]
                else:
                    padding = max_len_samples - waveform.shape[1]
                    waveform = torch.nn.functional.pad(waveform, (0, padding))
            return waveform

        except Exception as e:
            print(f"Warning: failed to load audio {full_path}: {e}")
            duration = self.max_length or 1.0
            num_samples = int(duration * self.sample_rate)
            return torch.zeros(1, num_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        row = self.df.iloc[idx]
        rel_path = row["path"]
        label = int(row["label"] if not pd.isna(row["label"]) else 0)

        waveform = self._load_audio(rel_path)
        try:
            if self.transform:
                waveform = self.transform(waveform)
        except Exception as e:
            print(f"Warning: transform failed for {rel_path}: {e}")
            # Return silent audio if transform fails
            num_samples = int(self.sample_rate * (self.max_length or 1.0))
            waveform = torch.zeros(1, num_samples)

        meta = {"path": rel_path, "index": int(idx)}
        return waveform, label, meta
