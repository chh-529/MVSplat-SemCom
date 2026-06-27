"""Plot SemCom (DeepJSCC) vs JPEG + ideal channel coding."""

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

JPEG = Path("outputs/test/jpeg")
PHASE2_CSV = Path("outputs/test/phase2_results.csv")
OUT = Path("outputs/test/jpeg_compare_figure.png")
CLEAN = 26.023


def load_jpeg():
    rows = []
    for sf in sorted(JPEG.glob("*/scores_all_avg.json")):
        m = re.match(r"jpeg_snr(-?\d+)", sf.parent.name)
        if not m:
            continue
        s = json.loads(sf.read_text())
        rows.append({"snr": int(m.group(1)), "psnr": s["psnr"], "tx_psnr": s["tx_psnr"]})
    return sorted(rows, key=lambda r: r["snr"])


def load_jscc():
    out = {"mse": [], "lpips": []}
    with PHASE2_CSV.open() as f:
        for r in csv.DictReader(f):
            if r["model"] == "full" and r["variant"] in ("mse", "lpips") and r["tx_psnr"]:
                out[r["variant"]].append(
                    {"snr": int(r["snr"]), "psnr": float(r["psnr"]), "tx_psnr": float(r["tx_psnr"])}
                )
    for k in out:
        out[k].sort(key=lambda r: r["snr"])
    return out


def main():
    jpeg = load_jpeg()
    jscc = load_jscc()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # (a) task PSNR vs SNR.
    ax = axes[0]
    ax.plot([r["snr"] for r in jpeg], [r["psnr"] for r in jpeg], "-D", color="#BA7517",
            label="JPEG + ideal coding (capacity)")
    ax.plot([r["snr"] for r in jscc["mse"]], [r["psnr"] for r in jscc["mse"]], "-o",
            color="#185FA5", label="DeepJSCC-MSE")
    ax.plot([r["snr"] for r in jscc["lpips"]], [r["psnr"] for r in jscc["lpips"]], "-s",
            color="#7F77DD", label="DeepJSCC-LPIPS")
    ax.axhline(CLEAN, color="gray", ls=":", label="clean (no channel)")
    ax.axvspan(-11, -3.5, color="#E24B4A", alpha=0.08)
    ax.text(-9.5, 14, "JPEG\noutage\n(cliff)", color="#A32D2D", fontsize=8, ha="center")
    ax.set_xlabel("channel SNR (dB)")
    ax.set_ylabel("rendered novel-view PSNR (dB)")
    ax.set_title("(a) Task quality vs SNR\n(JPEG-ideal strong at SNR>=0, cliffs below)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # (b) task PSNR vs transmission PSNR (matched-quality view).
    ax = axes[1]
    ax.plot([r["tx_psnr"] for r in jpeg], [r["psnr"] for r in jpeg], "-D", color="#BA7517", label="JPEG")
    ax.plot([r["tx_psnr"] for r in jscc["mse"]], [r["psnr"] for r in jscc["mse"]], "-o",
            color="#185FA5", label="DeepJSCC-MSE")
    ax.plot([r["tx_psnr"] for r in jscc["lpips"]], [r["psnr"] for r in jscc["lpips"]], "-s",
            color="#7F77DD", label="DeepJSCC-LPIPS")
    ax.axhline(CLEAN, color="gray", ls=":", label="clean")
    ax.set_xlabel("transmission-layer PSNR (context views, dB)")
    ax.set_ylabel("rendered novel-view PSNR (dB)")
    ax.set_title("(b) Task vs transmission quality\n(matched tx-PSNR -> similar task PSNR)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    fig.suptitle("DeepJSCC vs JPEG+ideal-coding (RE10K subset, MVSplat frozen, R=1/8)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT, dpi=130)
    print("saved", OUT)

    print("\n| SNR | JPEG tx | JPEG task | JSCC-MSE tx | JSCC-MSE task |")
    print("|---|---|---|---|---|")
    jm = {r["snr"]: r for r in jscc["mse"]}
    for r in jpeg:
        m = jm.get(r["snr"])
        mt = f"{m['tx_psnr']:.1f}" if m else "-"
        mk = f"{m['psnr']:.2f}" if m else "-"
        print(f"| {r['snr']} | {r['tx_psnr']:.1f} | {r['psnr']:.2f} | {mt} | {mk} |")


if __name__ == "__main__":
    main()
