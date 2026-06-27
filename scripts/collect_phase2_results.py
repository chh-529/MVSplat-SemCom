"""Aggregate Phase-2 sweep results into a table.

Reads outputs/test/phase2/*/scores_all_avg.json and prints a markdown table
plus per-model degradation (delta vs. that model's own clean run). Also writes
outputs/test/phase2_results.csv.

Usage: python scripts/collect_phase2_results.py
"""

import csv
import json
import re
from pathlib import Path

ROOT = Path("outputs/test/phase2")


def main():
    rows = []
    for score_file in sorted(ROOT.glob("*/scores_all_avg.json")):
        name = score_file.parent.name
        scores = json.loads(score_file.read_text())
        m = re.match(r"(full|base|wocv|wobbcrossattn)_(clean|(mse|lpips)_snr(\d+))", name)
        if m is None:
            continue
        model = m.group(1)
        if m.group(2) == "clean":
            variant, snr = "clean", None
        else:
            variant, snr = m.group(3), int(m.group(4))
        rows.append(
            {
                "model": model,
                "variant": variant,
                "snr": snr,
                "psnr": scores.get("psnr"),
                "ssim": scores.get("ssim"),
                "lpips": scores.get("lpips"),
                "tx_psnr": scores.get("tx_psnr"),
            }
        )

    if not rows:
        print("no results found under", ROOT)
        return

    clean = {r["model"]: r for r in rows if r["variant"] == "clean"}
    for r in rows:
        ref = clean.get(r["model"])
        r["d_psnr"] = (
            round(r["psnr"] - ref["psnr"], 2) if ref and r["variant"] != "clean" else None
        )

    model_order = {"full": 0, "base": 1, "wocv": 2, "wobbcrossattn": 3}
    rows.sort(key=lambda r: (model_order[r["model"]], r["variant"], r["snr"] or -1))

    out_csv = Path("outputs/test/phase2_results.csv")
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    header = ["model", "variant", "snr", "tx_psnr", "psnr", "d_psnr", "ssim", "lpips"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        cells = []
        for k in header:
            v = r.get(k)
            cells.append("" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        print("| " + " | ".join(cells) + " |")
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
