import os
import shap
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt

def run_shap_explainer(model, val_loader, device, save_dir="explainability", num_samples=10):
    """
    Run SHAP explainability on validation data.

    Args:
        model: Trained AASIST model.
        val_loader: DataLoader for validation/evaluation.
        device: torch.device ("cuda" or "cpu").
        save_dir: Folder for SHAP outputs.
        num_samples: Samples for background and explanation.
    """

    # 1️ Setup: output folder with timestamp
    timestamp = datetime.datetime.now().strftime("run_%Y_%m_%d_%H_%M")
    run_dir = os.path.join(save_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[SHAP] Outputs will be saved to {run_dir}")

    # 2️ Data selection: background/test batch acquisition
    model.eval()
    val_iter = iter(val_loader)
    try:
        background_batch, _ = next(val_iter)
    except StopIteration:
        val_iter = iter(val_loader)
        background_batch, _ = next(val_iter)
    try:
        test_batch, test_labels = next(val_iter)
    except StopIteration:
        val_iter = iter(val_loader)
        test_batch, test_labels = next(val_iter)

    background_batch = background_batch[:num_samples].to(device)
    test_batch = test_batch[:num_samples].to(device)
    test_labels = test_labels[:num_samples]

    # 3️ Model wrapper: handles tuple outputs (logits)
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            x = x.clone()
            output = self.model(x)
            if isinstance(output, tuple):
                return output[-1]  # last element = logits
            return output

    wrapped_model = ModelWrapper(model)

    # 4️ SHAP explainer
    try:
        explainer = shap.DeepExplainer(wrapped_model, background_batch)
    except Exception as e:
        print(f"[SHAP] DeepExplainer init failed ({e}), falling back to GradientExplainer...")
        explainer = shap.GradientExplainer(wrapped_model, background_batch)

    # 5️ SHAP value computation
    print("[SHAP] Computing SHAP values...")
    try:
        shap_values = explainer.shap_values(test_batch)
        shap_values = shap_values[1] if isinstance(shap_values, list) and len(shap_values) > 1 else shap_values
    except Exception as e:
        print(f"[SHAP] Computation failed: {e}")
        return None

    # 6️ Visualization per sample
    print("[SHAP] Saving visualizations...")
    for i in range(len(test_batch)):
        sample = test_batch[i].detach().cpu().numpy()
        shap_map = shap_values[i]

        # Squeeze and align shapes
        sample = np.squeeze(sample)
        shap_map = np.squeeze(shap_map)

        # --- 🔧 FIX: average over class dimension if 3D (e.g., [H, W, num_classes]) ---
        if shap_map.ndim == 3 and shap_map.shape[-1] == 2:
            shap_map = shap_map.mean(axis=-1)

        label = "bonafide" if int(test_labels[i].item()) == 0 else "spoof"

        plt.figure(figsize=(8, 4))
        plt.imshow(sample, cmap='gray', aspect='auto')
        plt.imshow(shap_map, cmap='seismic', alpha=0.6, aspect='auto')
        plt.title(f"Sample {i+1} - True: {label}")
        plt.colorbar(label='SHAP value (importance)')
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"sample_{i+1:03d}_{label}.png"))
        plt.close()

    # 7️ Summary plot
    try:
        print("[SHAP] Creating summary plot...")
        shap_vals_2d = shap_values.reshape(shap_values.shape[0], -1)
        test_batch_2d = test_batch.detach().cpu().numpy().reshape(test_batch.shape[0], -1)
        shap.summary_plot(shap_vals_2d, test_batch_2d, show=False)
        plt.savefig(os.path.join(run_dir, "shap_summary.png"))
        plt.close()
    except Exception as e:
        print(f"[SHAP] Summary plot skipped: {e}")

    print(f"[SHAP] Explainability completed. Visuals at: {run_dir}")
    return shap_values
