"""Feature-domain DeepJSCC (insertion point B).

Transmits MVSplat's per-view CNN features (128 x 64 x 64) instead of the raw
image. The sender runs the CNN backbone; the receiver gets reconstructed
features and continues with the cross-view transformer + cost volume.

Bandwidth is matched to the pixel-domain JSCC: the latent is channel_dim x 64 x
64 (= 12 x 64 x 64 for channel_dim=12), i.e. the same R = 1/8 relative to the
source image. The encoder therefore compresses 128 -> 12 channels (10.7x) at the
native 64x64 feature resolution (kept intact for the cost volume).
"""

from typing import Optional

import torch
from torch import Tensor, nn

from .deepjscc import DeepJSCCCfg, apply_channel, power_normalize


def _conv(in_ch, out_ch, act=True):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)]
    if act:
        layers.append(nn.PReLU())
    return nn.Sequential(*layers)


class FeatureJSCC(nn.Module):
    domain = "feature"

    def __init__(self, cfg: DeepJSCCCfg):
        super().__init__()
        self.cfg = cfg
        fc = cfg.feature_channels
        c = cfg.channel_dim
        # keep 64x64 spatial; compress channels fc -> c and back
        self.encoder = nn.Sequential(_conv(fc, 64), _conv(64, 32), _conv(32, c, act=False))
        self.decoder = nn.Sequential(_conv(c, 32), _conv(32, 64), _conv(64, fc, act=False))

    @property
    def bandwidth_ratio(self) -> float:
        # latent (c x 64 x 64) real values vs source image (3 x 256 x 256)
        return self.cfg.channel_dim / 96.0

    def sample_snr(self, batch: int, device: torch.device) -> Tensor:
        if self.cfg.snr_db is not None:
            return torch.full((batch,), float(self.cfg.snr_db), device=device)
        lo, hi = self.cfg.snr_min, self.cfg.snr_max
        return torch.rand(batch, device=device) * (hi - lo) + lo

    def forward(self, features: Tensor, snr_db: Optional[Tensor] = None) -> Tensor:
        """features: (..., C, H, W) with any leading batch/view dims."""
        batch_dims = features.shape[:-3]
        x = features.reshape(-1, *features.shape[-3:])
        if snr_db is None:
            snr_db = self.sample_snr(x.shape[0], x.device)
        z = power_normalize(self.encoder(x))
        y = apply_channel(z, snr_db, self.cfg.channel_type)
        x_hat = self.decoder(y)
        return x_hat.reshape(*batch_dims, *x_hat.shape[-3:])
