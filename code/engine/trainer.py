import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any
from ..utils.metrics import compute_metrics
from ..utils.checkpoint import save_checkpoint


def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion, optimizer, device: str) -> Dict[str, Any]:
    model.train()
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

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
        probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()

        all_preds.extend(preds)
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    acc, auc, precision, recall, f1 = compute_metrics(all_labels, all_preds, all_probs)
    return {
        "loss": epoch_loss,
        "accuracy": acc,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate(model: nn.Module, dataloader: DataLoader, criterion, device: str) -> Dict[str, Any]:
    model.eval()
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            images, labels, _ = batch
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    acc, auc, precision, recall, f1 = compute_metrics(all_labels, all_preds, all_probs)
    return {
        "loss": epoch_loss,
        "accuracy": acc,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


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

        # Create checkpoint state
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_metrics": val_metrics,
        }

        # Save last checkpoint
        save_checkpoint(state, is_best=False, output_dir=run_dir)

        # Save best if improved
        if val_metrics.get("auc", 0.0) > best_auc:
            best_auc = val_metrics["auc"]
            save_checkpoint(state, is_best=True, output_dir=run_dir)
            print(f"New best model with AUC {best_auc:.4f}")

    return best_auc
