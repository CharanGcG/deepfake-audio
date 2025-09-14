# --- code/models/backbones.py ---

from code.models.cvt import cvt_13, cvt_21, cvt_w24, HuggingFaceCvTBackbone
import torch.nn as nn

def _strip_head(model: nn.Module) -> nn.Module:
    if hasattr(model, "head"):
        model.head = nn.Identity()
    if hasattr(model, "fc"):
        model.fc = nn.Identity()
    return model


def create_backbone(name: str = "cvt_w24", pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name.startswith("hf_cvt"):
        model_name_map = {
            "hf_cvt_w24": "microsoft/cvt-w24-384",
            "hf_cvt_21": "microsoft/cvt-21-384",
            "hf_cvt_13": "microsoft/cvt-13-384"
        }
        hf_model_name = model_name_map.get(name, "microsoft/cvt-w24-384")
        return HuggingFaceCvTBackbone(model_name=hf_model_name, pretrained=pretrained)
    else:
        if name == "cvt_13" and cvt_13:
            return _strip_head(cvt_13(pretrained=pretrained))
        elif name == "cvt_21" and cvt_21:
            return _strip_head(cvt_21(pretrained=pretrained))
        elif name == "cvt_w24" and cvt_w24:
            return _strip_head(cvt_w24(pretrained=pretrained))
        else:
            import timm
            try:
                model = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
                return model
            except Exception as e:
                raise ValueError(f"Unknown backbone or failed to create: {name}. Error: {e}")
