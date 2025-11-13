import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import os
from aasist.models.AASIST import Model
from explainability.gradcam_explainer import GradCAM

# --- CONFIG ---
# fake
#AUDIO_PATH = r"C:\Charan Files\deepfake-audio\dataset\LA\LA\ASVspoof2019_LA_eval\flac\LA_E_1000147.flac"  

# real
AUDIO_PATH = r"C:\Charan Files\deepfake-audio\dataset\LA\LA\ASVspoof2019_LA_eval\flac\LA_E_5849185.flac"  


MODEL_PATH = r"C:\Charan Files\deepfake-audio\aasist\models\weights\AASIST.pth"
SAVE_DIR = "gradcam_single_sample"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- DEVICE ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# window=torch.ones(n_fft, device="cuda")

# --- MODEL CONFIG ---
model_config = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0]
}

# --- LOAD MODEL ---
model = Model(model_config)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# --- HELPER: Load single raw audio ---
def load_single_audio(audio_path, nb_samp=64600):
    waveform, sr = torchaudio.load(audio_path)  # shape [channels, L]
    # convert to mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # shape [1, L]
    # trim or pad
    if waveform.size(1) > nb_samp:
        waveform = waveform[:, :nb_samp]
    elif waveform.size(1) < nb_samp:
        pad_len = nb_samp - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))
    # remove extra channel dim: shape [1, nb_samp]
    waveform = waveform.squeeze(0)
    # add batch dim: shape [1, nb_samp]
    waveform = waveform.unsqueeze(0)
    return waveform

# --- LOAD AUDIO ---
batch_x = load_single_audio(AUDIO_PATH, nb_samp=model_config["nb_samp"]).to(device)
batch_x.requires_grad = True
print(f"Input shape for model: {batch_x.shape}")  # should be [1, 64600]

# --- GRAD-CAM ---
target_layer = model.encoder[-1][0].conv2
gradcam = GradCAM(model, target_layer)

# forward + backward
_, batch_out = model(batch_x)
score = batch_out[:, 1]  # spoof class
model.zero_grad()
score.sum().backward()

# generate CAM
cam = gradcam.generate_cam()[0]  # shape (H, W)

waveform_vis = batch_x[0].detach().cpu().float()  # shape [64600]
spec = torch.stft(waveform_vis, n_fft=512, hop_length=256, return_complex=True, center=False).abs()
spec = spec.numpy()
spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)



plt.figure(figsize=(10, 4))
plt.imshow(spec, aspect='auto', origin='lower', cmap='gray')
plt.imshow(cam, aspect='auto', origin='lower', cmap='jet', alpha=0.5)
plt.colorbar(label='Grad-CAM intensity')
plt.title(f"Grad-CAM overlay for {os.path.basename(AUDIO_PATH)}")
plt.xlabel("Time")
plt.ylabel("Frequency")

save_path = os.path.join(SAVE_DIR, os.path.splitext(os.path.basename(AUDIO_PATH))[0] + "_gradcam.png")
plt.savefig(save_path)
plt.show()
print(f"Grad-CAM overlay saved at: {save_path}")