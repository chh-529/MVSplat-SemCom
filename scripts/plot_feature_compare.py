"""Feature-domain SemCom vs traditional feature source coding (+ pixel context).
task-PSNR vs SNR on the 429-scene test set, all at R=1/8."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEST = Path("outputs/test")
OUT = TEST / "feature_compare_figure.png"
SNRS = [-10, -7, -5, -3, 0, 5, 10, 15, 20]
CLEAN = 26.59


def series(sub, fmt):
    out = []
    for s in SNRS:
        p = TEST / sub / fmt.format(s) / "scores_all_avg.json"
        if p.exists():
            out.append((s, json.loads(p.read_text())["psnr"]))
    return out


def main():
    feat_jscc = series("featjscc", "e2e_snr{}")        # feature JSCC, full e2e
    feat_only = series("featjscc", "featonly_snr{}")    # feature JSCC, frozen MVSplat
    feat_quant = series("featjscc", "quant_snr{}")      # traditional PCA+quant
    pixel_jscc = series("negsnr", "e2e_snr{}")          # pixel JSCC e2e (context)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for ser, sty, lab in [
        (feat_jscc, ("-^", "#1D9E75"), "Feature-JSCC (e2e, learned)"),
        (feat_only, ("-o", "#85B7EB"), "Feature-JSCC (train-JSCC-only)"),
        (feat_quant, ("-D", "#BA7517"), "Feature PCA+quant (traditional)"),
        (pixel_jscc, ("--s", "#888780"), "Pixel-JSCC (e2e, context)"),
    ]:
        if ser:
            ax.plot([s for s, _ in ser], [v for _, v in ser], sty[0], color=sty[1], label=lab)
    ax.axhline(CLEAN, color="gray", ls=":", label="clean upper bound")
    ax.set_xlabel("channel SNR (dB)")
    ax.set_ylabel("rendered novel-view PSNR (dB)")
    ax.set_title("Feature-domain: learned JSCC vs traditional source coding\n(429 test, R=1/8)")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT, dpi=130); plt.close(fig)
    print("saved", OUT)

    print("\n| SNR | feat-JSCC e2e | feat-JSCC only | feat PCA+quant | pixel-JSCC e2e |")
    print("|---|---|---|---|---|")
    d = {k: dict(v) for k, v in [("a", feat_jscc), ("b", feat_only), ("c", feat_quant), ("d", pixel_jscc)]}
    for s in SNRS:
        row = [f"{d[x][s]:.2f}" if s in d[x] else "-" for x in "abcd"]
        print(f"| {s} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
