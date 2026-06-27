# MVSplat + SemCom 專案總覽（修改清單 + 實驗清單 + 資料夾地圖）

> 本檔記錄：我們對原版 MVSplat repo 做了哪些改動、跑了哪些實驗、檔案放在哪裡。
> 架構/訓練細節見 `SEMCOM_NOTES.md`；逐步開發 log 見 `EXPERIMENTS.md`。

---

## A. 對原版 MVSplat repo 的修改

### A1. 新增的檔案（我們的程式碼，原 repo 沒有）

| 路徑 | 內容 |
|---|---|
| `src/model/semcom/deepjscc.py` | Pixel-domain DeepJSCC（5 層 conv，AWGN 通道）|
| `src/model/semcom/degradations.py` | 傳統壓縮 baseline（JPEG / noise / blur）|
| `src/model/semcom/feature_jscc.py` | Feature-domain JSCC（架構 B）|
| `src/model/semcom/feature_codec.py` | Feature-domain 傳統 codec（PCA+量化）|
| `src/model/semcom/__init__.py` | `get_semcom()` dispatch |
| `src/scripts/train_deepjscc.py` | JSCC 預訓練腳本 |
| `config/semcom/*.yaml` | deepjscc / degradation / feature_jscc / feature_quant |
| `scripts/*` | 所有實驗 sweep / 繪圖 / 資料下載腳本（見下方說明）|

### A2. 修改的原 repo 檔案（5 個 + 2 個 config）

| 檔案 | 改了什麼 |
|---|---|
| `src/config.py` | `RootCfg` 加 `semcom: Optional[SemComCfg]` |
| `src/main.py` | 建構 semcom 傳入 ModelWrapper；semcom 在場時 ckpt 用 `strict=False`；`limit_val_batches` |
| `src/model/model_wrapper.py` | `transmit_context_views`（pixel domain 過通道）；`configure_optimizers` param groups + `freeze_mvsplat`；feature semcom 掛到 encoder；`save_ply` 匯出 hook；`OptimizerCfg`/`TestCfg`/`TrainerCfg` 加欄位 |
| `src/model/encoder/encoder_costvolume.py` | 加 `self.feature_semcom`，傳入 backbone |
| `src/model/encoder/backbone/backbone_multiview.py` | CNN 後、Transformer 前套用 `feature_semcom`（插入點 B）|
| `config/main.yaml` | semcom 預設群組；optimizer `semcom_lr`/`freeze_mvsplat`；trainer `limit_val_batches`；test `save_ply` |

> 設計原則：所有改動都**向後相容**——不加 `+semcom=...` 時，原版 MVSplat 指令完全照舊運作。

---

## B. 我們做的實驗（依時間順序）

| # | 實驗 | 問題 | 主要結論 | 圖 |
|---|---|---|---|---|
| 1 | **JSCC 預訓練** | 影像傳輸 JSCC 能否收斂 | MSE 版 PSNR@10dB≈28.4、LPIPS≈25.1，收斂 | — |
| 2 | **Phase 2 假設驗證** | SemCom 退化傷哪個模組 | **cost volume 最脆弱**（拿掉它退化最少）| phase2_big_figure |
| 3 | **實驗 2（機制）** | 是「跨 view 不一致」還是「資訊損失」 | **反駁不一致假設**；真機制是高頻可匹配細節損失（blur 最傷）| exp2_figure |
| 4 | **JPEG 比較（pixel）** | SemCom vs 傳統壓縮 | 現 regime JPEG 勝；JSCC 優勢在斷崖區 | jpeg_compare_figure |
| 5 | **Phase 3 e2e（小資料）** | e2e 能否補回退化 | 39 場景下幾乎補不回（資料量假象）| phase3_figure |
| 6 | **Route C 資料擴充** | 取得更多 re10k | 自建 re10k_big（3000/500 場景）| — |
| 7 | **Phase 3 e2e（大資料）** | 同上、用正式資料 | **結論翻轉**：e2e 穩定補回 +0.24~+0.58 dB | phase3_big / old_vs_new |
| 8 | **負 SNR 研究** | 展示 JSCC 無斷崖優勢 | JSCC 在 [−10,20] 優雅退化 vs JPEG 斷崖 | task_vs_snr / tx_psnr_vs_snr |
| 9 | **tx vs task 分析** | tx PSNR 是否充分統計量 | 大致是；但 e2e 犧牲 tx 換 task（task-oriented 證據）| task_vs_tx_figure |
| 10 | **Feature-domain（架構 B）** | 傳特徵 vs 傳影像 | **Feature-JSCC 完勝**；且 feature domain 裡 learned>>traditional（與 pixel 相反）| feature_compare_figure |
| 11 | **PLY 匯出 / demo** | 把 3D 高斯球丟線上 viewer | `test.save_ply=true` 匯出標準 3DGS .ply | — |

---

## C. scripts/ 用途速查

| 類別 | 腳本 |
|---|---|
| 資料 | `download_re10k.py`（抓+轉）、`build_re10k_subset.py`、`extract_feature_stats.py`（PCA）|
| 訓練 orchestrator | `run_overnight.sh`（大資料全流程）、`run_negsnr.sh`、`run_featjscc.sh`、`run_feature_compare.sh` |
| 評測 sweep | `run_phase2_sweep.sh`、`run_phase3_eval.sh`、`run_jpeg_compare.sh`、`run_exp2_sweep.sh` |
| 繪圖 | `plot_phase2.py`、`plot_exp2.py`、`plot_jpeg_compare.py`、`plot_task_vs_snr.py`、`plot_tx_analysis.py`、`plot_feature_compare.py`、`collect_overnight.py`、`collect_phase2_results.py` |

---

## D. 重要產物位置（2026-06 整理後）

| 東西 | 位置 |
|---|---|
| 最終圖檔（11 張）| `results/figures/`（見該資料夾 README）|
| 大資料結果摘要 | `results/overnight_results.md` |
| 我們訓練的 JSCC 權重 | `checkpoints/jscc`（小資料）、`jscc_big`、`jscc_neg`（[−10,20]）|
| Feature PCA basis | `checkpoints/feature_pca.pt` |
| Phase 3 訓練的模型（最終）| `outputs/training/{phase3,phase3_big,phase3_neg,featjscc}/*/checkpoints/*step_20000*` |
| 評測分數（raw）| `outputs/test/<exp>/<run>/scores_all_avg.json`（按實驗分組）|
| Demo PLY + 輸入圖 / 影片 | `outputs/demo/demo_ply/`、`outputs/demo/demo_video/` |
| 所有訓練/評測 log | `outputs/logs/` |

### outputs/ 整理後結構
```
outputs/
├── test/        評測分數（17 個實驗分組）+ 圖檔     [plot 腳本依賴此路徑，勿移]
├── training/    訓練好的模型（只留 step_20000）
├── demo/        demo_ply + demo_video
└── logs/        所有 .log
```
> 註：中間步數 checkpoint（5k/10k/15k）已刪（從未使用；需要可重訓）。

