# code/models/backbones.py
"""
Backbone definitions.

Provides CvT variants and optionally timm-based models if available.
"""

import torch.nn as nn
import timm

from code.models.cvt import cvt_13, cvt_21, cvt_w24


def _strip_head(model: nn.Module) -> nn.Module:
    """Remove classification head if present, keep as pure feature extractor."""
    if hasattr(model, "head"):
        model.head = nn.Identity()
    if hasattr(model, "fc"):
        model.fc = nn.Identity()
    return model


def create_backbone(name: str = "cvt_13", pretrained: bool = False) -> nn.Module:
    """Factory to create backbone model.

    Args:
        name: backbone name ("cvt_13", "cvt_21", "cvt_w24", or timm model name)
        pretrained: whether to load pretrained weights if supported
    """
    name = name.lower()
    if name == "cvt_13" and cvt_13:
        model = cvt_13(pretrained=pretrained)
        return _strip_head(model)
    elif name == "cvt_21" and cvt_21:
        model = cvt_21(pretrained=pretrained)
        return _strip_head(model)
    elif name == "cvt_w24" and cvt_w24:
        model = cvt_w24(pretrained=pretrained)
        return _strip_head(model)
    else:
        # fallback to timm
        try:
            model = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
            return model
        except Exception as e:
            raise ValueError(f"Unknown backbone or failed to create: {name}. Error: {e}")

