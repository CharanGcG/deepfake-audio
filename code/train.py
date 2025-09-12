# code/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .args import get_args
from .dataset import DeepfakeDataset
from .transforms import get_train_transform, get_val_transform
from .models.classifier import DeepfakeModel
from .engine.trainer import run_phase
from .engine.evaluator import evaluate
from .utils.seed import seed_everything
from .utils.logger import create_logger


def main():
    args = get_args()
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = create_logger("train", args.run_dir)

    # Data
    train_ds = DeepfakeDataset(args.train_csv, args.root_dir, transform=get_train_transform())
    val_ds = DeepfakeDataset(args.val_csv, args.root_dir, transform=get_val_transform())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = DeepfakeModel(args.backbone, pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()

    # Phase 1: head-only
    model.freeze_backbone()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr_head)
    best_auc = 0.0
    best_auc = run_phase("Head", model, train_loader, val_loader, criterion, optimizer, None, device, args.epochs_head, args.run_dir, best_auc)

    # Phase 2: fine-tune
    model.unfreeze_backbone()
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr_backbone},
        {"params": model.classifier.parameters(), "lr": args.lr_head},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_finetune)
    best_auc = run_phase("Fine", model, train_loader, val_loader, criterion, optimizer, scheduler, device, args.epochs_finetune, args.run_dir, best_auc)

    logger.info(f"Training complete. Best AUC: {best_auc:.4f}")

    if args.do_cam:
        from .cam import run_gradcam
        run_gradcam(model, val_ds, device, args.run_dir)


if __name__ == "__main__":
    main()

