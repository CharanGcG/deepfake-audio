# code/engine/trainer.py
"""
Training loop utilities for deepfake detection.

Includes:
- train_one_epoch
- run_phase (head-only or fine-tune)

Logs losses/metrics each epoch and updates best checkpoint if validation improves.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any
from ..utils.metrics import compute_metrics
from ..utils.checkpoint import save_checkpoint


def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion, optimizer, device: str) -> Dict[str, Any]:
    model.train()
    running_loss = 0.0
    all_labels, all_preds = [], []

    for batch in dataloader:
        images, labels, _ = batch
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = epoch_loss
    return metrics


def run_phase(phase_name: str, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
              criterion, optimizer, scheduler, device: str, num_epochs: int,
              run_dir: str, best_auc: float) -> float:
    """Run one training phase (head-only or fine-tune)."""

    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)

        val_metrics = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                pass

        print(f"[{phase_name}] Epoch {epoch+1}/{num_epochs} | Train: {train_metrics} | Val: {val_metrics}")

        # Save last checkpoint
        save_checkpoint(model, optimizer, epoch, val_metrics, run_dir, is_best=False)

        # Save best if improved
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            save_checkpoint(model, optimizer, epoch, val_metrics, run_dir, is_best=True)
            print(f"New best model with AUC {best_auc:.4f}")

    return best_auc

