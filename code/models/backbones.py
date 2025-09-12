# code/models/backbones.py
"""
Backbone definitions.

Provides CvT variants and optionally timm-based models if available.
"""

import torch.nn as nn
import timm

try:
    from cvt import cvt_13, cvt_21, cvt_w24, CvT
except ImportError:
    cvt_13 = cvt_21 = cvt_w24 = None
    CvT = None


def create_backbone(name: str = "cvt_13", pretrained: bool = False) -> nn.Module:
    """Factory to create backbone model.

    Args:
        name: backbone name ("cvt_13", "cvt_21", "cvt_w24", or timm model name)
        pretrained: whether to load pretrained weights if supported
    """
    name = name.lower()
    if name == "cvt_13" and cvt_13:
        return cvt_13(pretrained=pretrained)
    elif name == "cvt_21" and cvt_21:
        return cvt_21(pretrained=pretrained)
    elif name == "cvt_w24" and cvt_w24:
        return cvt_w24(pretrained=pretrained)
    else:
        # fallback to timm
        try:
            model = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
            return model
        except Exception as e:
            raise ValueError(f"Unknown backbone or failed to create: {name}. Error: {e}")

