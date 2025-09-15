try:
    from code.models.cvt_ms.cls_cvt import ConvolutionalVisionTransformer
    _has_ms = True
except ImportError:
    _has_ms = False

print(_has_ms)