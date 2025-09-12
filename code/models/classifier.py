# code/models/classifier.py
"""
Classifier model that wraps backbone + classification head.

We assume binary classification (real vs fake).
"""

import torch
import torch.nn as nn
from .backbones import create_backbone


class DeepfakeModel(nn.Module):
    def __init__(self, backbone_name: str = "cvt_13", pretrained: bool = False, num_classes: int = 2):
        super().__init__()
        self.backbone = create_backbone(backbone_name, pretrained=pretrained)

        # Determine backbone output features
        if hasattr(self.backbone, "num_features"):
            in_features = self.backbone.num_features
        elif hasattr(self.backbone, "head") and hasattr(self.backbone.head, "in_features"):
            in_features = self.backbone.head.in_features
        else:
            # fallback - try with dummy forward
            try:
                import torch
                dummy = torch.zeros(1, 3, 256, 256)
                out = self.backbone(dummy)
                in_features = out.shape[1]
            except Exception:
                in_features = 768  # safe default

        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        logits = self.classifier(feats)
        return logits

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
