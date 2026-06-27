"""Stage-1 pretraining of DeepJSCC on RE10K frames (image transmission only).

Two variants:
  - mse:   loss = MSE                      (pixel-fidelity-oriented)
  - lpips: loss = 0.1 * MSE + LPIPS(vgg)   (perceptual/semantic-oriented)

Usage:
  python -m src.scripts.train_deepjscc --variant mse --channel_dim 12 --steps 30000
"""

import argparse
import json
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..model.semcom.deepjscc import DeepJSCC, DeepJSCCCfg


class RE10kFrames(Dataset):
    """All frames (as undecoded JPEG bytes) from the RE10K .torch chunks."""

    def __init__(self, chunk_dir: Path, crop: int = 256, train: bool = True,
                 max_frames: int | None = None):
        self.crop = crop
        self.train = train
        self.jpegs = []
        for chunk_path in sorted(chunk_dir.glob("*.torch")):
            for example in torch.load(chunk_path):
                self.jpegs.extend(example["images"])
            if max_frames is not None and len(self.jpegs) >= max_frames * 2:
                break  # stop loading once we have plenty to subsample from
        if max_frames is not None and len(self.jpegs) > max_frames:
            idx = torch.linspace(0, len(self.jpegs) - 1, max_frames).long().tolist()
            self.jpegs = [self.jpegs[i] for i in idx]
        print(f"{chunk_dir}: {len(self.jpegs)} frames")

    def __len__(self):
        return len(self.jpegs)

    def __getitem__(self, idx):
        image = Image.open(BytesIO(self.jpegs[idx].numpy().tobytes()))
        w, h = image.size
        c = self.crop
        scale = max(c / w, c / h)
        if scale > 1:
            image = image.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
            w, h = image.size
        if self.train:
            left = torch.randint(0, w - c + 1, ()).item()
            top = torch.randint(0, h - c + 1, ()).item()
        else:
            left, top = (w - c) // 2, (h - c) // 2
        image = image.crop((left, top, left + c, top + c)).convert("RGB")
        x = torch.from_numpy(np.array(image, copy=True))
        return x.permute(2, 0, 1).float() / 255.0


@torch.no_grad()
def evaluate(model, val_loader, device, snrs=(5.0, 10.0, 15.0)):
    model.eval()
    results = {}
    for snr in snrs:
        psnrs = []
        for x in val_loader:
            x = x.to(device)
            snr_t = torch.full((x.shape[0],), snr, device=device)
            x_hat = model(x, snr_db=snr_t)
            mse = ((x - x_hat) ** 2).flatten(1).mean(dim=1).clamp_min(1e-10)
            psnrs.append(-10 * mse.log10())
        results[snr] = torch.cat(psnrs).mean().item()
    model.train()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["mse", "lpips"], default="mse")
    parser.add_argument("--channel_dim", type=int, default=12)
    parser.add_argument("--snr_min", type=float, default=0.0)
    parser.add_argument("--snr_max", type=float, default=20.0)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_root", type=Path, default=Path("datasets/re10k"))
    parser.add_argument("--out_dir", type=Path, default=Path("checkpoints/jscc"))
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_frames", type=int, default=None,
                        help="cap (subsample) training frames to bound RAM")
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(0)

    train_set = RE10kFrames(args.data_root / "train", train=True, max_frames=args.max_frames)
    val_set = RE10kFrames(args.data_root / "test", train=False, max_frames=4096)
    # The val pass is only a progress probe; cap it to keep eval cheap.
    if len(val_set) > 512:
        val_set = torch.utils.data.Subset(
            val_set, torch.linspace(0, len(val_set) - 1, 512).long().tolist()
        )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(val_set, batch_size=32, num_workers=4)

    cfg = DeepJSCCCfg(
        name="deepjscc",
        channel_dim=args.channel_dim,
        snr_db=None,  # train across the SNR range
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        channel_type="awgn",
        weights=None,
        trainable=True,
    )
    model = DeepJSCC(cfg).to(device)

    lpips_fn = None
    if args.variant == "lpips":
        from lpips import LPIPS

        lpips_fn = LPIPS(net="vgg").to(device)
        lpips_fn.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.variant}_c{args.channel_dim}"
    log_path = args.out_dir / f"log_{tag}.jsonl"
    best_psnr10 = 0.0
    step = 0
    t0 = time.time()
    model.train()

    while step < args.steps:
        for x in train_loader:
            if step >= args.steps:
                break
            x = x.to(device, non_blocking=True)
            x_hat = model(x)
            mse = F.mse_loss(x_hat, x)
            if args.variant == "lpips":
                loss = 0.1 * mse + lpips_fn(x_hat, x, normalize=True).mean()
            else:
                loss = mse
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            if step % 200 == 0:
                psnr = -10 * mse.detach().clamp_min(1e-10).log10().item()
                print(
                    f"step {step}/{args.steps} loss {loss.item():.4f} "
                    f"train_psnr {psnr:.2f} ({(time.time() - t0) / step:.2f}s/it)",
                    flush=True,
                )

            if step % args.eval_every == 0 or step == args.steps:
                val = evaluate(model, val_loader, device)
                entry = {"step": step, "variant": args.variant, **{f"psnr@{k}": v for k, v in val.items()}}
                print("eval:", json.dumps(entry), flush=True)
                with log_path.open("a") as f:
                    f.write(json.dumps(entry) + "\n")
                payload = {
                    "state_dict": model.state_dict(),
                    "args": vars(args) | {"data_root": str(args.data_root), "out_dir": str(args.out_dir)},
                    "step": step,
                    "val_psnr": val,
                }
                torch.save(payload, args.out_dir / f"jscc_{tag}_last.pt")
                if val[10.0] > best_psnr10:
                    best_psnr10 = val[10.0]
                    torch.save(payload, args.out_dir / f"jscc_{tag}.pt")

    print(f"done. best psnr@10dB = {best_psnr10:.2f}")


if __name__ == "__main__":
    main()
