# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .args import get_args
from .dataset import DeepfakeDataset
from .transforms import get_train_transform, get_val_transform
from .models.classifier import DeepfakeModel
from .engine.trainer import run_phase
from .utils.seed import seed_everything
from .utils.logger import get_logger
from .utils.checkpoint import save_checkpoint, load_checkpoint


def main():
    args = get_args()
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = get_logger("train", args.run_dir)
    logger.info(f"Started Training with args: {args}")

    # Data
    logger.info("Loading training dataset...")
    train_ds = DeepfakeDataset(args.train_csv, args.root_dir, transform=get_train_transform())
    logger.info(f"Training dataset loaded with {len(train_ds)} samples")

    logger.info("Loading validation dataset...")
    val_ds = DeepfakeDataset(args.val_csv, args.root_dir, transform=get_val_transform())
    logger.info(f"Validation dataset loaded with {len(val_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    logger.info("DataLoaders created")

    # Model
    logger.info(f"Building model with backbone: {args.backbone}, pretrained={args.pretrained}")
    model = DeepfakeModel(args.backbone, pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    logger.info("Model and criterion initialized")

    best_auc = 0.0

    # Resume from checkpoint if provided
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        ckpt = load_checkpoint(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        best_auc = ckpt.get("best_auc", 0.0)
        logger.info(f"Resumed with best AUC: {best_auc:.4f}")

    # Phase 1: head-only
    if args.epochs_head > 0:
        logger.info("Phase 1: Head-only training started")
        model.freeze_backbone()
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr_head)

        best_auc = run_phase(
            "Head", model, train_loader, val_loader, criterion, optimizer, None,
            device, args.epochs_head, args.run_dir, best_auc
        )

        save_checkpoint({
            "model": model.state_dict(),
            "best_auc": best_auc,
            "args": vars(args)
        }, is_best=True, output_dir=args.run_dir, filename="head_last.pth")

        logger.info(f"Phase 1 complete. Best AUC: {best_auc:.4f}")

    # Phase 2: fine-tune
    if args.epochs_finetune > 0:
        logger.info("Phase 2: Fine-tuning started")
        model.unfreeze_backbone()
        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": args.lr_backbone},
            {"params": model.classifier.parameters(), "lr": args.lr_head},
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_finetune)

        best_auc = run_phase(
            "Fine", model, train_loader, val_loader, criterion, optimizer, scheduler,
            device, args.epochs_finetune, args.run_dir, best_auc
        )

        save_checkpoint({
            "model": model.state_dict(),
            "best_auc": best_auc,
            "args": vars(args)
        }, is_best=True, output_dir=args.run_dir, filename="finetune_last.pth")

        logger.info(f"Phase 2 complete. Best AUC: {best_auc:.4f}")

    logger.info("Training finished successfully")

    if args.do_cam:
        logger.info("Running GradCAM visualization...")
        from .cam import run_gradcam
        run_gradcam(model, val_ds, device, args.run_dir)
        logger.info("GradCAM visualization complete")


if __name__ == "__main__":
    logger_main = get_logger("outputs/run_debug", name="main")
    logger_main.info("Starting main training script")
    main()
