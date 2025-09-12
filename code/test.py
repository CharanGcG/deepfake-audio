# code/test.py
import torch
from torch.utils.data import DataLoader
from .args import get_args
from .dataset import DeepfakeDataset
from .transforms import get_val_transform
from .models.classifier import DeepfakeModel
from .engine.evaluator import evaluate
import torch.nn as nn


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    test_ds = DeepfakeDataset(args.test_csv, args.root_dir, transform=get_val_transform())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = DeepfakeModel(args.backbone, pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss()

    checkpoint_path = f"{args.run_dir}/best.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    metrics = evaluate(model, test_loader, criterion, device)
    print("Test metrics:", metrics)


if __name__ == "__main__":
    main()

