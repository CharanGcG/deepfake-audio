# --- code/models/classifier.py ---

from code.models.backbones import create_backbone
import torch
import torch.nn as nn

class DeepfakeModel(nn.Module):
    def __init__(self, backbone_name: str = "hf_cvt_w24", pretrained: bool = True, num_classes: int = 2):
        super().__init__()
        self.backbone = create_backbone(backbone_name, pretrained=pretrained)

        if backbone_name.startswith("hf_cvt"):
            in_features_map = {
                "hf_cvt_w24": 384,
                "hf_cvt_21": 256,
                "hf_cvt_13": 64
            }
            in_features = in_features_map.get(backbone_name, 384)
        else:
            if hasattr(self.backbone, "num_features"):
                in_features = self.backbone.num_features
            else:
                try:
                    dummy = torch.zeros(1, 3, 224, 224)
                    out = self.backbone(dummy)
                    if isinstance(out, (list, tuple)):
                        out = out[-1]
                    if out.dim() > 2:
                        out = out.mean(dim=[2,3])
                    in_features = out.shape[1]
                except Exception:
                    in_features = 768

        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        if feats.dim() > 2:
            feats = feats.mean(dim=[2,3])
        logits = self.classifier(feats)
        return logits

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
