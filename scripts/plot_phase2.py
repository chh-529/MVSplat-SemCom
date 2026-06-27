"""Plot Phase-2 hypothesis-verification results from phase2_results.csv."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = Path("outputs/test/phase2_results.csv")
OUT = Path("outputs/test/phase2_figure.png")

MODEL_LABEL = {
    "full": "full (cost-vol + refine)",
    "base": "base (cost-vol, no refine)",
    "wocv": "w/o cost volume",
    "wobbcrossattn": "w/o cross-view attn",
}
MODEL_COLOR = {"full": "#185FA5", "base": "#1D9E75", "wocv": "#D85A30", "wobbcrossattn": "#7F77DD"}
VAR_STYLE = {"mse": "-o", "lpips": "--s"}


def load():
    rows = []
    with CSV.open() as f:
        for r in csv.DictReader(f):
            for k in ("snr", "psnr", "ssim", "lpips", "tx_psnr", "d_psnr"):
                r[k] = float(r[k]) if r[k] not in ("", None) else None
            rows.append(r)
    return rows


def get(rows, model, variant):
    pts = [r for r in rows if r["model"] == model and r["variant"] == variant]
    pts.sort(key=lambda r: r["snr"])
    return pts


def main():
    rows = load()
    clean = {r["model"]: r["psnr"] for r in rows if r["variant"] == "clean"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Panel (a): downstream PSNR vs SNR, full model, both variants + clean ref.
    ax = axes[0]
    for variant in ("mse", "lpips"):
        pts = get(rows, "full", variant)
        ax.plot([p["snr"] for p in pts], [p["psnr"] for p in pts], VAR_STYLE[variant],
                color=MODEL_COLOR["full"], label=f"full, {variant.upper()}-JSCC")
    ax.axhline(clean["full"], color="gray", ls=":", label="full, clean (no channel)")
    ax.set_xlabel("channel SNR (dB)")
    ax.set_ylabel("rendered novel-view PSNR (dB)")
    ax.set_title("(a) Task quality vs channel SNR")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel (b): module vulnerability — degradation Δ vs SNR for all 4 models (MSE).
    ax = axes[1]
    for model in ("full", "base", "wobbcrossattn", "wocv"):
        pts = get(rows, model, "mse")
        ax.plot([p["snr"] for p in pts], [p["d_psnr"] for p in pts], "-o",
                color=MODEL_COLOR[model], label=MODEL_LABEL[model])
    ax.set_xlabel("channel SNR (dB)")
    ax.set_ylabel("PSNR degradation Δ vs own clean (dB)")
    ax.set_title("(b) Which module is vulnerable (MSE-JSCC)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel (c): hypothesis scatter — transmission PSNR vs task PSNR (full model).
    ax = axes[2]
    for variant in ("mse", "lpips"):
        pts = get(rows, "full", variant)
        ax.plot([p["tx_psnr"] for p in pts], [p["psnr"] for p in pts], VAR_STYLE[variant],
                color=MODEL_COLOR["full"], label=f"{variant.upper()}-JSCC")
        for p in pts:
            ax.annotate(f"{int(p['snr'])}", (p["tx_psnr"], p["psnr"]),
                        fontsize=6, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("transmission-layer PSNR (context views, dB)")
    ax.set_ylabel("rendered novel-view PSNR (dB)")
    ax.set_title("(c) Transmission vs task quality (full)\nlabels = SNR")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "MVSplat + DeepJSCC Phase-2: hypothesis verification (RE10K 38-scene subset, MVSplat frozen)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT, dpi=130)
    print("saved", OUT)


if __name__ == "__main__":
    main()
