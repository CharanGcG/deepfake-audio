# --- code/models/cvt.py ---
"""
CvT models with optional pretrained weights (via timm or Hugging Face).
"""

import torch
import torch.nn as nn

try:
    from transformers import CvtForImageClassification, CvtModel
    _has_hf = True
except ImportError:
    _has_hf = False


class ConvEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, kernel_size=7, stride=4, padding=2):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size, stride, padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (H, W)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class CvT(nn.Module):
    def __init__(self, img_size=256, in_channels=3, num_classes=2, embed_dim=64, depth=6, num_heads=4):
        super().__init__()
        self.embed = ConvEmbedding(in_channels, embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x, _ = self.embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(1)
        return self.head(x)


class HuggingFaceCvTBackbone(nn.Module):
    def __init__(self, model_name="microsoft/cvt-w24-384-22k", pretrained=True, return_features="hidden_state"):
        """
        return_features: "logits" | "hidden_state"
        """
        super().__init__()
        if not _has_hf:
            raise ImportError("Transformers library is required for HuggingFaceCvTBackbone.")
        if return_features == "hidden_state":
            # use base model to get last_hidden_state
            self.model = CvtModel.from_pretrained(model_name) if pretrained \
                         else CvtModel.from_config(model_name)
            self._use_logits = False
        else:
            # use classification heads for logits
            self.model = CvtForImageClassification.from_pretrained(model_name) if pretrained \
                         else CvtForImageClassification.from_config(model_name)
            # strip off classifier
            self.model.classifier = nn.Identity()
            self._use_logits = True
        self.return_features = return_features

    def forward(self, x):
        if self._use_logits:
            outputs = self.model(pixel_values=x)
            # logits shape: (batch, num_classes)
            return outputs.logits
        else:
            outputs = self.model(pixel_values=x)
            # last_hidden_state: (batch, seq_len, hidden_dim)
            feat = outputs.last_hidden_state
            # mean-pool
            return feat.mean(dim=1)


# Factory functions for CvT variants
def cvt_13(pretrained=False, **kwargs):
    if _has_hf:
        return HuggingFaceCvTBackbone("microsoft/cvt-13-384-22k", pretrained=pretrained, **kwargs)
    return CvT(embed_dim=64, depth=6, num_heads=4, **kwargs)


def cvt_21(pretrained=False, **kwargs):
    if _has_hf:
        return HuggingFaceCvTBackbone("microsoft/cvt-21-384-22k", pretrained=pretrained, **kwargs)
    return CvT(embed_dim=128, depth=12, num_heads=8, **kwargs)


def cvt_w24(pretrained=False, **kwargs):
    if _has_hf:
        return HuggingFaceCvTBackbone("microsoft/cvt-w24-384-22k", pretrained=pretrained, **kwargs)
    return CvT(embed_dim=384, depth=24, num_heads=12, **kwargs)
