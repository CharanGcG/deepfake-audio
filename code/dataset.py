# code/dataset.py
"""
DeepfakeDataset

Reads a CSV produced by the Kaggle dataset and yields (image_tensor, label, meta)
meta can include the relative path and index for logging/troubleshooting.

Assumptions:
- CSV contains columns: 'path' and 'label' (0 for fake, 1 for real)
- ``root_dir`` is the parent directory where paths in CSV are relative to

Robustness:
- If an image file is missing or unreadable, returns a zero tensor and logs a warning
- Supports optional transform callable (PIL -> Tensor)
"""

from typing import Optional, Callable, Tuple, Dict, Any
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import torch


class DeepfakeDataset(Dataset):
    def __init__(self, csv_path: str, root_dir: str, transform: Optional[Callable] = None, img_size: int = 256):
        """Create dataset from CSV.

        Args:
            csv_path: Path to CSV file (must contain 'path' and 'label')
            root_dir: Base directory for image paths in CSV
            transform: Optional callable applied to PIL.Image; must return torch.Tensor
            img_size: Fallback image size when creating synthetic zero images
        """
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        if "path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("CSV must contain 'path' and 'label' columns")

        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size

        # Reset index for safe integer indexing
        self.df = self.df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, rel_path: str) -> Image.Image:
        full_path = os.path.join(self.root_dir, rel_path)
        if not os.path.isfile(full_path):
            # Missing file -> create black image
            # Returning PIL image for compatibility with transforms
            # Using RGB
            print(f"Warning: image not found: {full_path}")
            return Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))
        try:
            img = Image.open(full_path).convert("RGB")
            return img
        except Exception as e:
            print(f"Warning: failed to open image {full_path}: {e}")
            return Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        row = self.df.iloc[idx]
        rel_path = row["path"]
        label = int(row["label"]) if not pd.isna(row["label"]) else 0

        img = self._load_image(rel_path)

        if self.transform is not None:
            try:
                img = self.transform(img)
            except Exception as e:
                # If transform fails, return zero tensor
                print(f"Warning: transform failed for {rel_path}: {e}")
                img = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
        else:
            # Convert to tensor manually (simple fallback)
            img = self._pil_to_tensor(img)

        meta = {"path": rel_path, "index": int(idx)}
        return img, label, meta

    @staticmethod
    def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
        # Basic conversion (no normalization)
        arr = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        # PIL stores as (W,H,RGB) bytes; convert to HWC then CHW
        arr = arr.reshape(img.size[1], img.size[0], 3)
        arr = arr.permute(2, 0, 1).float().div(255.0)
        return arr

