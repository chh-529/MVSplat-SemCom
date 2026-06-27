"""Synthetic context-view degradations for the mechanism experiment (Exp 2).

These are *not* learned channels; they are controlled distortions used to test
*why* SemCom hurts MVSplat. The key contrast is matched per-view distortion
magnitude with the cross-view consistency toggled:

  - kind=noise, correlation=independent : each view gets its own noise draw
  - kind=noise, correlation=shared      : all views of a scene share one draw
  - kind=blur                           : same Gaussian blur on every view
                                          (spatially coherent, preserves edge
                                           locations that cross-view matching
                                           relies on)

All modes take and return images in [0, 1] with shape (..., 3, H, W). The
image-level transmission PSNR is logged by ModelWrapper.transmit_context_views,
so each strength setting self-reports where it lands on the PSNR axis.
"""

import math
from dataclasses import dataclass
from io import BytesIO
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn


@dataclass
class DegradationCfg:
    name: Literal["degradation"]
    kind: Literal["noise", "blur", "jpeg"]
    # noise: std in [0,1]; blur: Gaussian sigma (px); jpeg: target bits-per-pixel
    strength: float
    correlation: Literal["independent", "shared"]  # only used for kind=noise


class ContextDegradation(nn.Module):
    def __init__(self, cfg: DegradationCfg):
        super().__init__()
        self.cfg = cfg

    def _gaussian_blur(self, images: Tensor) -> Tensor:
        sigma = self.cfg.strength
        if sigma <= 0:
            return images
        radius = max(1, int(math.ceil(3 * sigma)))
        ksize = 2 * radius + 1
        coords = torch.arange(ksize, device=images.device, dtype=images.dtype) - radius
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        kernel = (g[:, None] * g[None, :]).view(1, 1, ksize, ksize)
        c = images.shape[-3]
        kernel = kernel.expand(c, 1, ksize, ksize)
        flat = images.reshape(-1, c, *images.shape[-2:])
        blurred = F.conv2d(flat, kernel, padding=radius, groups=c)
        return blurred.reshape(images.shape)

    def _jpeg(self, images: Tensor) -> Tensor:
        """Classical separation baseline: JPEG-compress each view to <= target
        bpp (largest quality factor that fits the bit budget). If even quality 1
        exceeds the budget the link is in outage -> deliver a gray frame (total
        loss), which is what produces the classical cliff effect."""
        *lead, c, h, w = images.shape
        budget_bytes = self.cfg.strength * h * w / 8.0
        flat = images.reshape(-1, c, h, w)
        out = torch.empty_like(flat)
        for i in range(flat.shape[0]):
            arr = (flat[i].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
            pil = Image.fromarray(arr)
            best, lo, hi = None, 1, 95
            while lo <= hi:
                mid = (lo + hi) // 2
                buf = BytesIO()
                pil.save(buf, format="JPEG", quality=mid)
                if buf.tell() <= budget_bytes:
                    best, lo = mid, mid + 1
                else:
                    hi = mid - 1
            if best is None:
                dec = np.full_like(arr, 128)  # outage
            else:
                buf = BytesIO()
                pil.save(buf, format="JPEG", quality=best)
                buf.seek(0)
                dec = np.array(Image.open(buf).convert("RGB"))
            out[i] = torch.from_numpy(dec).float().div(255).permute(2, 0, 1).to(images.device)
        return out.reshape(images.shape)

    def forward(self, images: Tensor) -> Tensor:
        if self.cfg.kind == "blur":
            return self._gaussian_blur(images).clamp(0, 1)
        if self.cfg.kind == "jpeg":
            return self._jpeg(images)

        # kind == "noise"
        std = self.cfg.strength
        if self.cfg.correlation == "shared":
            # Share one noise draw across the view dim (images: b, v, c, h, w).
            assert images.dim() == 5, "shared noise expects (b, v, c, h, w)"
            b, v = images.shape[:2]
            base = torch.randn(b, 1, *images.shape[2:], device=images.device, dtype=images.dtype)
            noise = base.expand(b, v, *images.shape[2:]) * std
        else:
            noise = torch.randn_like(images) * std
        return (images + noise).clamp(0, 1)
