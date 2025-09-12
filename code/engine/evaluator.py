# code/engine/evaluator.py
"""
Evaluation utilities (validation & test).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any
from ..utils.metrics import compute_metrics


def evaluate(model: nn.Module, dataloader: DataLoader, criterion, device: str) -> Dict[str, Any]:
    model.eval()
    running_loss = 0.0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for batch in dataloader:
            images, labels, _ = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = epoch_loss
    return metrics
