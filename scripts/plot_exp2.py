"""Collect + plot Experiment 2 (mechanism: cross-view consistency)."""

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP2 = Path("outputs/test/exp2")
PHASE2_CSV = Path("outputs/test/phase2_results.csv")
OUT = Path("outputs/test/exp2_figure.png")


def load_exp2():
    rows = []
    for sf in sorted(EXP2.glob("*/scores_all_avg.json")):
        name = sf.parent.name
        s = json.loads(sf.read_text())
        rec = {"name": name, "psnr": s.get("psnr"), "tx_psnr": s.get("tx_psnr"),
               "ssim": s.get("ssim"), "lpips": s.get("lpips")}
        if (m := re.match(r"noise_(independent|shared)_std([\d.]+)", name)):
            rec["cond"], rec["param"] = f"noise_{m.group(1)}", float(m.group(2))
        elif (m := re.match(r"blur_sigma([\d.]+)", name)):
            rec["cond"], rec["param"] = "blur", float(m.group(1))
        else:
            continue
        rows.append(rec)
    return rows


def load_phase2_full():
    """Full-model JSCC points for overlay."""
    out = {"jscc_mse": [], "jscc_lpips": []}
    if not PHASE2_CSV.exists():
        return out
    with PHASE2_CSV.open() as f:
        for r in csv.DictReader(f):
            if r["model"] != "full" or r["variant"] not in ("mse", "lpips"):
                continue
            if not r["tx_psnr"]:
                continue
            out[f"jscc_{r['variant']}"].append((float(r["tx_psnr"]), float(r["psnr"])))
    for k in out:
        out[k].sort()
    return out


STYLE = {
    "noise_independent": ("#E24B4A", "-o", "noise, independent (inconsistent)"),
    "noise_shared": ("#1D9E75", "-o", "noise, shared (consistent)"),
    "blur": ("#888780", "-^", "blur (coherent)"),
    "jscc_mse": ("#185FA5", "--s", "JSCC-MSE"),
    "jscc_lpips": ("#7F77DD", "--s", "JSCC-LPIPS"),
}


def curve(rows, cond):
    pts = sorted([(r["tx_psnr"], r["psnr"]) for r in rows if r["cond"] == cond])
    return [p[0] for p in pts], [p[1] for p in pts]


def main():
    rows = load_exp2()
    jscc = load_phase2_full()
    clean = 26.023  # full-model clean reference

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # Panel (a): the controlled toggle — noise independent vs shared (+ blur).
    ax = axes[0]
    for cond in ("noise_independent", "noise_shared", "blur"):
        x, y = curve(rows, cond)
        c, ls, lab = STYLE[cond]
        ax.plot(x, y, ls, color=c, label=lab)
    ax.axhline(clean, color="gray", ls=":", lw=1, label="clean (no distortion)")
    ax.set_xlabel("transmission-layer PSNR (context views, dB)")
    ax.set_ylabel("rendered novel-view PSNR (dB)")
    ax.set_title("(a) Same magnitude, consistency toggled\n(indep ≈ shared; blur far worse)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel (b): everything overlaid incl. JSCC, to place the mechanism in context.
    ax = axes[1]
    for cond in ("noise_independent", "noise_shared", "blur"):
        x, y = curve(rows, cond)
        c, ls, lab = STYLE[cond]
        ax.plot(x, y, ls, color=c, label=lab)
    for key in ("jscc_mse", "jscc_lpips"):
        if jscc[key]:
            c, ls, lab = STYLE[key]
            ax.plot([p[0] for p in jscc[key]], [p[1] for p in jscc[key]], ls, color=c, label=lab)
    ax.axhline(clean, color="gray", ls=":", lw=1, label="clean")
    ax.set_xlabel("transmission-layer PSNR (context views, dB)")
    ax.set_ylabel("rendered novel-view PSNR (dB)")
    ax.set_title("(b) Mechanism vs learned JSCC")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Exp 2: distortion type dominates, cross-view consistency barely matters "
                 "(RE10K subset, MVSplat frozen, full model)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT, dpi=130)
    print("saved", OUT)

    # CSV dump + a quick matched-magnitude delta table.
    with Path("outputs/test/exp2_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "cond", "param", "tx_psnr", "psnr", "ssim", "lpips"])
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["cond"], r["param"])):
            w.writerow({k: r.get(k) for k in w.fieldnames})

    print("\nmatched-std noise: independent vs shared (task PSNR)")
    print("| std | tx_indep | task_indep | tx_shared | task_shared | shared-indep |")
    print("|---|---|---|---|---|---|")
    by_std = {}
    for r in rows:
        if r["cond"].startswith("noise_"):
            by_std.setdefault(r["param"], {})[r["cond"]] = r
    for std in sorted(by_std):
        i = by_std[std].get("noise_independent")
        s = by_std[std].get("noise_shared")
        if i and s:
            print(f"| {std} | {i['tx_psnr']:.2f} | {i['psnr']:.2f} | "
                  f"{s['tx_psnr']:.2f} | {s['psnr']:.2f} | {s['psnr'] - i['psnr']:+.2f} |")


if __name__ == "__main__":
    main()
