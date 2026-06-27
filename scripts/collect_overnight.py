"""Collect the overnight big-dataset results into figures + a markdown summary.
Safe to run repeatedly; skips anything missing."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEST = Path("outputs/test")
SUMMARY = Path("outputs/overnight_results.md")
SNRS = [0, 5, 10, 15, 20]
MODELS = ["full", "base", "wocv", "wobbcrossattn"]
COL = {"full": "#185FA5", "base": "#1D9E75", "wocv": "#D85A30", "wobbcrossattn": "#7F77DD"}


def sc(sub, name):
    p = TEST / sub / name / "scores_all_avg.json"
    return json.loads(p.read_text()) if p.exists() else None


def phase2_fig(clean):
    rows = []
    for m in MODELS:
        for v in ("mse", "lpips"):
            for snr in SNRS:
                s = sc("phase2_big", f"{m}_{v}_snr{snr}")
                if s:
                    rows.append(dict(model=m, variant=v, snr=snr, tx=s.get("tx_psnr"),
                                     psnr=s["psnr"], d=round(s["psnr"] - clean.get(m, float("nan")), 2),
                                     ssim=s["ssim"], lpips=s["lpips"]))
    if not rows:
        return rows
    with (TEST / "phase2_big_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for v, ls in (("mse", "-o"), ("lpips", "--s")):
        pts = sorted([r for r in rows if r["model"] == "full" and r["variant"] == v], key=lambda r: r["snr"])
        if pts:
            ax[0].plot([r["snr"] for r in pts], [r["psnr"] for r in pts], ls,
                       color="#185FA5", label=f"full {v.upper()}-JSCC")
    if "full" in clean:
        ax[0].axhline(clean["full"], color="gray", ls=":", label="clean")
    ax[0].set_title("(a) full model: task PSNR vs SNR")
    ax[0].set_xlabel("channel SNR (dB)"); ax[0].set_ylabel("novel-view PSNR (dB)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    for m in MODELS:
        pts = sorted([r for r in rows if r["model"] == m and r["variant"] == "mse"], key=lambda r: r["snr"])
        if pts:
            ax[1].plot([r["snr"] for r in pts], [r["d"] for r in pts], "-o", color=COL[m], label=m)
    ax[1].set_title("(b) degradation by model (MSE-JSCC)")
    ax[1].set_xlabel("channel SNR (dB)"); ax[1].set_ylabel("PSNR degradation Δ (dB)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.suptitle("Phase 2 re-eval on 500 test scenes (low variance)")
    fig.tight_layout(); fig.savefig(TEST / "phase2_big_figure.png", dpi=130); plt.close(fig)
    return rows


def phase3_fig(clean):
    p3 = {tag: {snr: sc("phase3_big", f"{tag}_snr{snr}") for snr in SNRS} for tag in ("frozen", "jscconly", "e2e")}
    p3 = {tag: {s: v["psnr"] for s, v in d.items() if v} for tag, d in p3.items()}
    if not any(p3.values()):
        return p3
    fig, ax = plt.subplots(figsize=(7, 5))
    sty = {"frozen": ("#888780", "-o", "frozen baseline"),
           "jscconly": ("#1D9E75", "-s", "train JSCC only"),
           "e2e": ("#185FA5", "-^", "full end-to-end")}
    for tag, d in p3.items():
        if d:
            xs = sorted(d)
            ax.plot(xs, [d[s] for s in xs], sty[tag][1], color=sty[tag][0], label=sty[tag][2])
    if "full" in clean:
        ax.axhline(clean["full"], color="gray", ls=":", label="clean upper bound")
    ax.set_xlabel("channel SNR (dB)"); ax.set_ylabel("rendered novel-view PSNR (dB)")
    ax.set_title("Phase 3 on big data (3000 train): e2e vs frozen")
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(TEST / "phase3_big_figure.png", dpi=130); plt.close(fig)
    return p3


def old_vs_new_fig():
    """The conclusion flip: e2e improvement over frozen, 39 vs 3000 scenes."""
    def delta(sub, frozen_tag):
        out = {}
        for snr in SNRS:
            fr = sc(sub, f"{frozen_tag}_snr{snr}")
            e2 = sc(sub, f"e2e_snr{snr}")
            if fr and e2:
                out[snr] = e2["psnr"] - fr["psnr"]
        return out
    # old run: outputs/test/phase3/{e2e,jscconly}; frozen baseline there was the
    # Phase-2 full-mse sweep (outputs/test/phase2 full_mse).
    old = {}
    for snr in SNRS:
        fr = sc("phase2", f"full_mse_snr{snr}")
        e2 = sc("phase3", f"e2e_snr{snr}")
        if fr and e2:
            old[snr] = e2["psnr"] - fr["psnr"]
    new = delta("phase3_big", "frozen")
    if not new:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    if old:
        ax.plot(sorted(old), [old[s] for s in sorted(old)], "-o", color="#D85A30",
                label="39 train scenes (old)")
    ax.plot(sorted(new), [new[s] for s in sorted(new)], "-^", color="#185FA5",
            label="3000 train scenes (new)")
    ax.axhline(0, color="gray", ls=":", label="no change vs frozen")
    ax.set_xlabel("channel SNR (dB)"); ax.set_ylabel("e2e improvement over frozen (dB)")
    ax.set_title("Conclusion flip: data size decides whether e2e helps")
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(TEST / "phase3_old_vs_new_figure.png", dpi=130); plt.close(fig)


def main():
    clean = {m: s["psnr"] for m in MODELS if (s := sc("phase2_big", f"{m}_clean"))}
    p2 = phase2_fig(clean)
    p3 = phase3_fig(clean)
    old_vs_new_fig()

    L = ["# Overnight big-dataset results (re10k_big: 3000 train / 500 test)\n"]
    if p3:
        L += ["## Phase 3: e2e vs frozen (big data)\n",
              "| SNR | frozen | train-JSCC-only | full e2e | e2e Δ |",
              "|---|---|---|---|---|"]
        for snr in SNRS:
            fr = p3.get("frozen", {}).get(snr); b = p3.get("jscconly", {}).get(snr); c = p3.get("e2e", {}).get(snr)
            if fr is not None and c is not None:
                bcell = f"{b:.2f}" if b is not None else "-"
                L.append(f"| {snr} | {fr:.2f} | {bcell} | {c:.2f} | {c-fr:+.2f} |")
        L.append(f"\nclean upper bound (full) = {clean.get('full', float('nan')):.2f}\n")
    if p2:
        cf = clean.get("full", float("nan"))
        L += ["## Phase 2 re-eval (full model, 500 test)\n",
              "| SNR | MSE task | MSE Δ | LPIPS task | LPIPS Δ |", "|---|---|---|---|---|"]
        for snr in SNRS:
            m = next((r for r in p2 if r["model"]=="full" and r["variant"]=="mse" and r["snr"]==snr), None)
            l = next((r for r in p2 if r["model"]=="full" and r["variant"]=="lpips" and r["snr"]==snr), None)
            if m:
                L.append(f"| {snr} | {m['psnr']:.2f} | {m['psnr']-cf:+.2f} | "
                         + (f"{l['psnr']:.2f} | {l['psnr']-cf:+.2f} |" if l else "- | - |"))
    L += ["\n## Figures",
          "- outputs/test/phase3_big_figure.png — e2e vs frozen (big data)",
          "- outputs/test/phase3_old_vs_new_figure.png — the conclusion flip (39 vs 3000 scenes)",
          "- outputs/test/phase2_big_figure.png — low-variance Phase 2"]
    SUMMARY.write_text("\n".join(L))
    print(f"wrote {SUMMARY}\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
