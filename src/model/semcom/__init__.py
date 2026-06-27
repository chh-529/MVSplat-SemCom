from typing import Optional, Union

import torch
from torch import nn

from .deepjscc import DeepJSCC, DeepJSCCCfg
from .degradations import ContextDegradation, DegradationCfg
from .feature_codec import FeatureCodec
from .feature_jscc import FeatureJSCC

# DeepJSCCCfg listed first so dacite resolves a JSCC dict to it; a degradation
# dict lacks the JSCC-required fields and falls through to DegradationCfg.
SemComCfg = Union[DeepJSCCCfg, DegradationCfg]


def get_semcom(cfg: Optional[SemComCfg]) -> Optional[nn.Module]:
    if cfg is None:
        return None

    if cfg.name == "degradation":
        print(f"==> Context degradation: kind={cfg.kind} strength={cfg.strength} "
              f"correlation={cfg.correlation}")
        return ContextDegradation(cfg)

    # DeepJSCC: pixel domain (before encoder) or feature domain (inside encoder)
    if cfg.domain == "feature":
        semcom = FeatureCodec(cfg) if cfg.method == "pca_quant" else FeatureJSCC(cfg)
    else:
        semcom = DeepJSCC(cfg)
    if cfg.weights is not None:
        ckpt = torch.load(cfg.weights, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        semcom.load_state_dict(state_dict)
        print(f"==> Loaded DeepJSCC weights from {cfg.weights}")
    else:
        print("==> DeepJSCC initialized from scratch (no weights given)")
    if not cfg.trainable:
        semcom.requires_grad_(False)
        semcom.eval()
    return semcom
