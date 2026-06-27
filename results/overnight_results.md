# Overnight big-dataset results (re10k_big: 3000 train / 500 test)

## Phase 3: e2e vs frozen (big data)

| SNR | frozen | train-JSCC-only | full e2e | e2e Δ |
|---|---|---|---|---|
| 0 | 22.50 | 22.51 | 23.08 | +0.58 |
| 5 | 23.61 | 23.56 | 24.00 | +0.39 |
| 10 | 24.07 | 23.97 | 24.36 | +0.30 |
| 15 | 24.21 | 24.10 | 24.47 | +0.26 |
| 20 | 24.25 | 24.15 | 24.49 | +0.24 |

clean upper bound (full) = 26.59

## Phase 2 re-eval (full model, 500 test)

| SNR | MSE task | MSE Δ | LPIPS task | LPIPS Δ |
|---|---|---|---|---|
| 0 | 22.52 | -4.07 | 22.04 | -4.55 |
| 5 | 23.58 | -3.01 | 22.76 | -3.83 |
| 10 | 24.04 | -2.55 | 23.00 | -3.59 |
| 15 | 24.18 | -2.41 | 23.07 | -3.52 |
| 20 | 24.21 | -2.38 | 23.09 | -3.50 |

## Figures
- outputs/test/phase3_big_figure.png — e2e vs frozen (big data)
- outputs/test/phase3_old_vs_new_figure.png — the conclusion flip (39 vs 3000 scenes)
- outputs/test/phase2_big_figure.png — low-variance Phase 2