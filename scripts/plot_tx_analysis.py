"""Transmission-layer analysis on the test set (DeepJSCC trained over [-10,20]):
 (A) tx_PSNR vs SNR for JPEG vs JSCC-MSE (frozen + full e2e).
 (B) task_PSNR vs tx_PSNR scatter -- tests whether transmission PSNR is a
     sufficient statistic for task quality. e2e sits above the frozen+JPEG curve
     because it trades pixel fidelity for task-useful information."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEST = Path("outputs/test")
SNRS = [-10, -7, -5, -3, 0, 5, 10, 15, 20]
CLEAN = 26.59


def series(sub, fmt):
    out = []
    for s in SNRS:
        p = TEST / sub / fmt.format(s) / "scores_all_avg.json"
        if p.exists():
            d = json.loads(p.read_text())
            out.append((s, d.get("tx_psnr"), d.get("psnr")))
    return out


def main():
    jpeg = series("jpeg_big", "jpeg_snr{}")
    frozen = series("negsnr", "frozen_snr{}")
    e2e = series("negsnr", "e2e_snr{}")

    # ---- Figure A: tx_PSNR vs SNR ----
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot([s for s, _, _ in jpeg], [tx for _, tx, _ in jpeg], "-D", color="#BA7517",
            label="JPEG + ideal coding")
    ax.plot([s for s, _, _ in frozen], [tx for _, tx, _ in frozen], "-o", color="#185FA5",
            label="DeepJSCC-MSE frozen")
    ax.plot([s for s, _, _ in e2e], [tx for _, tx, _ in e2e], "-^", color="#1D9E75",
            label="DeepJSCC-MSE full e2e")
    cliff = [s for s, tx, _ in jpeg if tx < 15]
    if cliff:
        ax.axvspan(-11, max(cliff) + 1, color="#E24B4A", alpha=0.07)
        ax.text(-9.5, 14, "JPEG\noutage", color="#A32D2D", fontsize=9)
    ax.set_xlabel("channel SNR (dB)")
    ax.set_ylabel("transmission-layer PSNR (context views, dB)")
    ax.set_title("(A) Receiver-side reconstruction quality vs SNR")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(TEST / "tx_psnr_vs_snr_figure.png", dpi=130); plt.close(fig)

    # ---- Figure B: task vs tx (exclude JPEG outage) ----
    jpeg_ok = [(s, tx, k) for s, tx, k in jpeg if tx >= 15]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot([tx for _, tx, _ in frozen], [k for _, _, k in frozen], "-o", color="#185FA5",
            label="DeepJSCC-MSE frozen", ms=7)
    ax.plot([tx for _, tx, _ in e2e], [k for _, _, k in e2e], "-^", color="#1D9E75",
            label="DeepJSCC-MSE full e2e", ms=7)
    ax.plot([tx for _, tx, _ in jpeg_ok], [k for _, _, k in jpeg_ok], "-D", color="#BA7517",
            label="JPEG", ms=6)
    ax.axhline(CLEAN, color="gray", ls=":", label="clean upper bound")
    ax.annotate("frozen & JPEG lie on one curve\n→ tx PSNR predicts task",
                xy=(27.5, 23.8), xytext=(30, 22.3), fontsize=8.5, color="#444441",
                arrowprops=dict(arrowstyle="->", color="#888780"))
    ax.annotate("e2e sits above the curve\n→ trades tx fidelity for task-useful info",
                xy=(24.9, 24.2), xytext=(22, 25.3), fontsize=8.5, color="#0F6E56",
                arrowprops=dict(arrowstyle="->", color="#1D9E75"))
    ax.set_xlabel("transmission-layer PSNR (context views, dB)")
    ax.set_ylabel("rendered novel-view PSNR (dB)")
    ax.set_title("(B) Task vs transmission quality\nis tx PSNR a sufficient statistic?")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(TEST / "task_vs_tx_figure.png", dpi=130); plt.close(fig)

    print("saved tx_psnr_vs_snr_figure.png and task_vs_tx_figure.png")
    print("\n| method | SNR | tx_PSNR | task_PSNR |")
    print("|---|---|---|---|")
    for lab, ser in (("MSE-frozen", frozen), ("MSE-e2e", e2e), ("JPEG", jpeg)):
        for s, tx, k in ser:
            print(f"| {lab} | {s} | {tx:.2f} | {k:.2f} |")


if __name__ == "__main__":
    main()
