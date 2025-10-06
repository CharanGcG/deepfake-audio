import os
import torch
from torch.utils.data import DataLoader
import torchaudio
import matplotlib.pyplot as plt
from code.dataset import AudioDataset  # adjust import if needed

# === Config ===
csv_file = r"C:\Charan Files\deepfake-audio\dataset\metadata_csv\train.csv"
root_dir = r"C:\Charan Files\deepfake-audio\dataset"

sample_rate = 16000
max_length = 4  # seconds
batch_size = 4

# Optional: log-mel transform
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_mels=64,
    n_fft=1024,
    hop_length=256
)

# Initialize dataset
dataset = AudioDataset(
    csv_path=csv_file,
    root_dir=root_dir,
    transform=mel_transform,
    target_sample_rate=sample_rate,
    max_length=max_length
)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === Basic info ===
print(f"Number of samples: {len(dataset)}")

# Fetch a single sample
waveform, label, meta = dataset[0]
print(f"Sample 0 - Waveform shape: {waveform.shape}, Label: {label}, Path: {meta['path']}")

# Fetch multiple samples
for i in range(3):
    waveform, label, meta = dataset[i]
    print(f"Sample {i}: Shape={waveform.shape}, Label={label}, Path={meta['path']}")

# === Iterate a batch ===
for batch_waveforms, batch_labels, batch_meta in loader:
    print(f"Batch waveform shape: {batch_waveforms.shape}")
    print(f"Batch labels: {batch_labels}")
    print(f"Batch metadata: {batch_meta}")
    break  # only first batch

# === Visualize a few samples ===
for i in range(2):
    waveform, label, meta = dataset[i]
    
    # If mel-spectrogram transform is applied
    if len(waveform.shape) == 3:  # [channels, n_mels, time]
        spec = waveform.squeeze(0)  # remove channel dim -> [n_mels, time]
        
        # Plot mean waveform over mel bins
        plt.figure(figsize=(10, 2))
        plt.plot(spec.mean(dim=0).numpy())
        plt.title(f"Waveform (mean over mel bins) - Sample {i} - Label {label}")
        plt.show()

        # Plot mel spectrogram
        plt.figure(figsize=(10, 4))
        plt.imshow(spec.log2().detach().numpy(), origin='lower', aspect='auto', cmap='magma')
        plt.title(f"Mel Spectrogram - Sample {i} - Label {label}")
        plt.colorbar()
        plt.show()
    else:  # raw waveform [1, N]
        plt.figure(figsize=(10, 2))
        plt.plot(waveform.squeeze(0).numpy())
        plt.title(f"Raw Waveform - Sample {i} - Label {label}")
        plt.show()