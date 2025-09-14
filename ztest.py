import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA version (from PyTorch):", torch.version.cuda)
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
