"""Task-PSNR vs SNR: DeepJSCC vs JPEG across SNR -10..20 on the 429-scene test set.
The task-layer analogue of Figure A (which showed transmission-layer PSNR).
JSCC is trained over [-10,20] so it is valid in the cliff region. The e2e line is
added once outputs/test/negsnr/e2e_snr*/ exists."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEST = Path("outputs/test")
OUT = TEST / "task_vs_snr_figure.png"
SNRS = [-10, -7, -5, -3, 0, 5, 10, 15, 20]
CLEAN = 26.59


def task_series(sub, fmt):
    out = []
    for s in SNRS:
        p = TEST / sub / fmt.format(s) / "scores_all_avg.json"
        if p.exists():
            out.append((s, json.loads(p.read_text())["psnr"]))
    return out


def main():
    jpeg = task_series("jpeg_big", "jpeg_snr{}")
    frozen = task_series("negsnr", "frozen_snr{}")
    e2e = task_series("negsnr", "e2e_snr{}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    if jpeg:
        ax.plot([s for s, _ in jpeg], [v for _, v in jpeg], "-D", color="#BA7517",
                label="JPEG + ideal coding")
    if frozen:
        ax.plot([s for s, _ in frozen], [v for _, v in frozen], "-o", color="#185FA5",
                label="DeepJSCC-MSE frozen")
    if e2e:
        ax.plot([s for s, _ in e2e], [v for _, v in e2e], "-^", color="#1D9E75",
                label="DeepJSCC-MSE full e2e")
    ax.axhline(CLEAN, color="gray", ls=":", label="clean upper bound")

    cliff = [s for s, v in jpeg if v < 15]
    if cliff:
        ax.axvspan(-11, max(cliff) + 1, color="#E24B4A", alpha=0.07)
        ax.text(-9.5, 13, "JPEG\noutage", color="#A32D2D", fontsize=9)

    ax.set_xlabel("channel SNR (dB)")
    ax.set_ylabel("rendered novel-view PSNR (dB)")
    ax.set_title("Task quality vs SNR (429 test, R=1/8)\n"
                 "DeepJSCC degrades gracefully where JPEG cliffs")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT, dpi=130); plt.close(fig)
    print("saved", OUT)

    print("\n| SNR | JPEG | JSCC frozen | JSCC e2e |")
    print("|---|---|---|---|")
    dj, df, de = dict(jpeg), dict(frozen), dict(e2e)
    for s in SNRS:
        print(f"| {s} | {dj.get(s, '-') if isinstance(dj.get(s), str) else (f'{dj[s]:.2f}' if s in dj else '-')} "
              f"| {f'{df[s]:.2f}' if s in df else '-'} | {f'{de[s]:.2f}' if s in de else '-'} |")


if __name__ == "__main__":
    main()
