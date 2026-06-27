"""Traditional (non-learned) feature source codec: the feature-domain analogue
of the JPEG+ideal-coding baseline.

Transform coding on MVSplat's CNN features: PCA over the 128 channels, keep the
top-P components, uniform-scalar-quantize each to Q bits, and assume ideal
capacity-achieving channel coding. The bit budget at SNR is the same as the
pixel-domain JPEG baseline (same bandwidth, R=1/8):
    B = k * log2(1 + SNR_lin),  k = 24576 complex uses per view.
P is chosen to fill the budget at Q bits/coefficient; if even P=1 does not fit,
the link is in outage and the receiver falls back to the feature mean.
"""

import math
from typing import Optional

import torch
from torch import Tensor, nn

from .deepjscc import DeepJSCCCfg

K_CHANNEL_USES = 24576  # complex uses per view (matches pixel JSCC / JPEG)
SPATIAL = 64 * 64


class FeatureCodec(nn.Module):
    domain = "feature"

    def __init__(self, cfg: DeepJSCCCfg):
        super().__init__()
        self.cfg = cfg
        pca = torch.load(cfg.pca_path, map_location="cpu")
        # registered as buffers so they move with the module / load with state
        self.register_buffer("mean", pca["mean"])               # [128]
        self.register_buffer("components", pca["components"])    # [128,128] top->low
        self.register_buffer("comp_std", pca["comp_std"])        # [128]

    def _budget_components(self, snr_db: float) -> int:
        B = K_CHANNEL_USES * math.log2(1 + 10 ** (snr_db / 10.0))
        return int(B // (SPATIAL * self.cfg.quant_bits))

    def forward(self, features: Tensor, snr_db: Optional[Tensor] = None) -> Tensor:
        snr = self.cfg.snr_db if self.cfg.snr_db is not None else float(self.cfg.snr_max)
        P = min(self.components.shape[0], self._budget_components(snr))
        batch_dims = features.shape[:-3]
        c, h, w = features.shape[-3:]
        x = features.reshape(-1, c, h, w).permute(0, 2, 3, 1).reshape(-1, c)  # [N,128]

        xc = x - self.mean
        if P < 1:  # outage -> deliver the mean feature
            x_hat = self.mean.expand_as(x)
        else:
            comp = self.components[:P]                 # [P,128]
            coeffs = xc @ comp.T                       # [N,P]
            # uniform scalar quantization to Q bits over [-3sigma, 3sigma]
            levels = 2 ** self.cfg.quant_bits
            rng = 3.0 * self.comp_std[:P]
            step = (2 * rng) / (levels - 1)
            q = torch.round((coeffs.clamp(-rng, rng) + rng) / step)
            coeffs_hat = q * step - rng
            x_hat = coeffs_hat @ comp + self.mean      # [N,128]

        out = x_hat.reshape(*features.reshape(-1, c, h, w).shape[:1], h, w, c)
        out = out.permute(0, 3, 1, 2).reshape(*batch_dims, c, h, w)
        return out
