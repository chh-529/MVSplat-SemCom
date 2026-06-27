"""Deep joint source-channel coding (DeepJSCC) for transmitting context views.

Architecture follows the original DeepJSCC (Bourtsoulatze et al., 2019) as
implemented in https://github.com/chunbaobao/Deep-JSCC-PyTorch: a 5-layer
convolutional encoder/decoder with PReLU activations, per-sample power
normalization, and an AWGN (or Rayleigh, with perfect-CSI equalization)
channel in between.

With two stride-2 layers, the latent is C x H/4 x W/4 real values, i.e.
k = C * (H/4) * (W/4) / 2 complex channel symbols for n = 3 * H * W source
symbols, giving a bandwidth ratio R = k / n = C / 96.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


@dataclass
class DeepJSCCCfg:
    name: Literal["deepjscc"]
    channel_dim: int  # C; bandwidth ratio = C / 96
    snr_db: Optional[float]  # None = sample uniformly from [snr_min, snr_max]
    snr_min: float
    snr_max: float
    channel_type: Literal["awgn", "rayleigh"]
    weights: Optional[str]  # path to Stage-1 pretrained weights
    trainable: bool  # False = freeze (Phase 2)
    domain: Literal["pixel", "feature"] = "pixel"  # feature -> insertion point B
    feature_channels: int = 128  # MVSplat d_feature (feature domain only)
    method: Literal["jscc", "pca_quant"] = "jscc"  # feature domain: learned vs traditional
    quant_bits: int = 2  # bits/coefficient for the traditional PCA codec
    pca_path: str = "checkpoints/feature_pca.pt"


class _ConvPReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=stride, padding=2)
        self.act = nn.PReLU()
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="leaky_relu")
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.conv(x))


class _DeconvPReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, sigmoid: bool = False):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=5,
            stride=stride,
            padding=2,
            output_padding=stride - 1,
        )
        self.act = nn.Sigmoid() if sigmoid else nn.PReLU()
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="leaky_relu")
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.conv(x))


class JSCCEncoder(nn.Module):
    def __init__(self, channel_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            _ConvPReLU(3, 16, stride=2),
            _ConvPReLU(16, 32, stride=2),
            _ConvPReLU(32, 32),
            _ConvPReLU(32, 32),
            _ConvPReLU(32, channel_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class JSCCDecoder(nn.Module):
    def __init__(self, channel_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            _DeconvPReLU(channel_dim, 32),
            _DeconvPReLU(32, 32),
            _DeconvPReLU(32, 32),
            _DeconvPReLU(32, 16, stride=2),
            _DeconvPReLU(16, 3, stride=2, sigmoid=True),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.layers(z)


def power_normalize(z: Tensor) -> Tensor:
    """Normalize each sample to unit average power per (real) symbol."""
    flat = z.flatten(start_dim=1)
    norm = flat.norm(dim=1, keepdim=True) + 1e-8
    scale = flat.shape[1] ** 0.5 / norm
    return z * scale.view(-1, *[1] * (z.dim() - 1))


def apply_channel(
    z: Tensor,
    snr_db: Tensor,
    channel_type: str,
) -> Tensor:
    """AWGN: y = z + n.  Rayleigh assumes perfect CSI + equalization, which is
    equivalent to AWGN with the noise scaled by 1/|h| per sample."""
    noise_std = (10.0 ** (-snr_db / 10.0)).sqrt()
    if channel_type == "rayleigh":
        # |h|^2 ~ Exp(1) for h ~ CN(0, 1)
        h_sq = torch.empty_like(noise_std).exponential_(1.0).clamp_min(1e-4)
        noise_std = noise_std / h_sq.sqrt()
    noise_std = noise_std.view(-1, *[1] * (z.dim() - 1))
    return z + noise_std * torch.randn_like(z)


class DeepJSCC(nn.Module):
    """Transmit images through a learned analog channel code.

    Input/output: float images in [0, 1] with shape (..., 3, H, W); any number
    of leading batch dims (e.g. batch x view) is supported.
    """

    domain = "pixel"

    def __init__(self, cfg: DeepJSCCCfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = JSCCEncoder(cfg.channel_dim)
        self.decoder = JSCCDecoder(cfg.channel_dim)

    @property
    def bandwidth_ratio(self) -> float:
        return self.cfg.channel_dim / 96.0

    def sample_snr(self, batch: int, device: torch.device) -> Tensor:
        if self.cfg.snr_db is not None:
            return torch.full((batch,), float(self.cfg.snr_db), device=device)
        lo, hi = self.cfg.snr_min, self.cfg.snr_max
        return torch.rand(batch, device=device) * (hi - lo) + lo

    def forward(self, images: Tensor, snr_db: Optional[Tensor] = None) -> Tensor:
        batch_dims = images.shape[:-3]
        x = images.reshape(-1, *images.shape[-3:])
        if snr_db is None:
            snr_db = self.sample_snr(x.shape[0], x.device)
        z = power_normalize(self.encoder(x))
        y = apply_channel(z, snr_db, self.cfg.channel_type)
        x_hat = self.decoder(y)
        return x_hat.reshape(*batch_dims, *x_hat.shape[-3:])
