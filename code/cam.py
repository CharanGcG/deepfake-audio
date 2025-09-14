"""
GradCAM utility to visualize explanations.
"""
import os
import torch
import numpy as np
from torchvision.utils import save_image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from code.models.classifier import DeepfakeModel

def run_gradcam(model, dataset, device: str, run_dir: str, num_samples: int = 10):
    model.eval()
    model.to(device)

    # For CvT backbone, select convolutional embedding layer for GradCAM
    target_layers = [model.backbone.embed.proj]
    cam = GradCAM(model=model, target_layers=target_layers)

    out_dir = os.path.join(run_dir, "cam")
    os.makedirs(out_dir, exist_ok=True)

    for i in range(min(num_samples, len(dataset))):
        img, label, meta = dataset[i]
        input_tensor = img.unsqueeze(0).to(device)

        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(label)])
        grayscale_cam = grayscale_cam[0, :]

        rgb_img = np.transpose(img.cpu().numpy(), (1, 2, 0))
        rgb_img = np.clip(rgb_img, 0, 1)

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        safe_path = meta['path'].replace('/', '_').replace('\\', '_')  # Properly escaped
        save_path = os.path.join(out_dir, f"cam_{i}_{safe_path}.png")

        save_image(torch.tensor(cam_image).permute(2, 0, 1).float() / 255.0, save_path)
        print(f"Saved GradCAM image: {save_path}")
