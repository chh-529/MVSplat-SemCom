"""Fit a PCA basis on MVSplat's CNN features, for the traditional feature codec.

Runs the (pretrained) CNN backbone on a sample of RE10K images, collects the
128-d per-location feature vectors, and saves the PCA mean / components / per-
component std to checkpoints/feature_pca.pt. Runs on CPU so it does not disturb
GPU training.
"""

import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.model.encoder.backbone import BackboneMultiview


def load_backbone():
    bb = BackboneMultiview(feature_channels=128, downscale_factor=4,
                           no_cross_attn=False, use_epipolar_trans=False)
    sd = torch.load("checkpoints/re10k.ckpt", map_location="cpu")["state_dict"]
    prefix = "encoder.backbone."
    bb_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = bb.load_state_dict(bb_sd, strict=False)
    print(f"backbone loaded ({len(bb_sd)} keys; {len(missing)} missing)")
    return bb.eval()


def sample_images(data_root: Path, n: int, crop: int = 256):
    chunks = sorted((data_root / "train").glob("*.torch"))
    imgs = []
    for cp in chunks:
        for ex in torch.load(cp):
            jpg = ex["images"][len(ex["images"]) // 2]  # middle frame
            im = Image.open(BytesIO(jpg.numpy().tobytes())).convert("RGB")
            w, h = im.size
            im = im.crop(((w - crop) // 2, (h - crop) // 2,
                          (w - crop) // 2 + crop, (h - crop) // 2 + crop))
            imgs.append(torch.from_numpy(np.array(im)).float().div(255).permute(2, 0, 1))
            if len(imgs) >= n:
                return torch.stack(imgs)
    return torch.stack(imgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, default=Path("datasets/re10k_big"))
    ap.add_argument("--n_images", type=int, default=48)
    ap.add_argument("--out", type=Path, default=Path("checkpoints/feature_pca.pt"))
    args = ap.parse_args()

    bb = load_backbone()
    imgs = sample_images(args.data_root, args.n_images)  # [N,3,256,256]
    print(f"running CNN on {imgs.shape[0]} images (CPU)...")
    feats = []
    with torch.no_grad():
        for i in range(0, imgs.shape[0], 8):
            batch = imgs[i:i + 8].unsqueeze(1)  # [b,1,3,256,256]
            fl = bb.extract_feature(bb.normalize_images(batch))
            f = torch.stack([x[0] for x in fl], dim=1)  # [b,1,128,64,64]
            feats.append(f.reshape(-1, 128))  # [(b*64*64),128]
    X = torch.cat(feats)  # [M,128]
    print(f"collected {X.shape[0]} feature vectors")

    mean = X.mean(0)
    Xc = X - mean
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(cov)          # ascending
    components = evecs.flip(1).T                    # [128,128], rows = top->low
    coeffs = Xc @ components.T                       # [M,128] in PCA space
    comp_std = coeffs.std(0)                         # per-component std (for quant range)

    torch.save({"mean": mean, "components": components, "comp_std": comp_std,
                "explained_var": evals.flip(0)}, args.out)
    ev = evals.flip(0)
    print(f"saved {args.out}")
    print(f"top-12 components explain {ev[:12].sum() / ev.sum() * 100:.1f}% of variance")
    print(f"top-32 components explain {ev[:32].sum() / ev.sum() * 100:.1f}%")


if __name__ == "__main__":
    main()
