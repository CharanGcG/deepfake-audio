import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # register hooks
        self.forward_hook = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, target_class=None):
        if self.activations is None or self.gradients is None:
            raise ValueError("Forward and backward passes must be run before generating CAM.")

        # Compute channel-wise weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Compute CAM
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))

        # Normalize to [0,1]
        B, _, H, W = cam.shape
        cam_np = cam.cpu().numpy()
        cam_norm = np.zeros_like(cam_np)
        for i in range(B):
            cam_min = cam_np[i].min()
            cam_max = cam_np[i].max()
            cam_norm[i] = (cam_np[i] - cam_min) / (cam_max - cam_min + 1e-8)

        return cam_norm.squeeze(1)

    def __del__(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def run_gradcam_explainer(model, val_loader, device, save_dir="gradcam_overlay", target_class=1, freq_bins=257, time_frames=250):
    os.makedirs(save_dir, exist_ok=True)

    target_layer = model.encoder[-1][0].conv2
    gradcam = GradCAM(model, target_layer)

    model.to(device)
    model.eval()

    for idx, (batch_x, utt_ids) in enumerate(val_loader):
        batch_x = batch_x.to(device)
        batch_x.requires_grad = True

        _, batch_out = model(batch_x)
        score = batch_out[:, target_class]

        model.zero_grad()
        score.sum().backward()

        cam_maps = gradcam.generate_cam(target_class=target_class)

        for i, utt_id in enumerate(utt_ids):
            cam = cam_maps[i]
            spec = batch_x[i].detach().cpu().numpy()

            if spec.ndim == 1:
                spec = spec.reshape(freq_bins, time_frames)
            elif spec.ndim == 3:
                spec = spec.squeeze(0)

            spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)

            plt.figure(figsize=(10, 4))
            plt.imshow(spec, aspect='auto', origin='lower', cmap='gray')
            plt.imshow(cam, aspect='auto', origin='lower', cmap='jet', alpha=0.5)
            plt.colorbar(label='Grad-CAM intensity')
            plt.title(f"Grad-CAM overlay for {utt_id}")
            plt.xlabel("Time")
            plt.ylabel("Frequency")
            save_path = os.path.join(save_dir, f"{utt_id}_gradcam_overlay.png")
            plt.savefig(save_path)
            plt.close()

    print(f"Grad-CAM overlay visualizations saved in {save_dir}")