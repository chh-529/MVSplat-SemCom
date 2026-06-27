# MVSplat + SemCom 實驗記錄

研究假設：DeepJSCC 傳輸 context views 時，各 view 失真彼此獨立、且（perceptual 訓練時）偏向保留語意而非像素精確度，可能破壞 MVSplat 的 cross-view attention 與 cost volume matching。

## 新增的程式

| 檔案 | 用途 |
|---|---|
| `src/model/semcom/deepjscc.py` | DeepJSCC encoder/decoder + power norm + AWGN/Rayleigh 通道（架構參考 chunbaobao/Deep-JSCC-PyTorch）|
| `src/model/semcom/__init__.py` | `get_semcom()`：依 config 建構、載權重、凍結 |
| `config/semcom/deepjscc.yaml` | Hydra config group，用 `+semcom=deepjscc` 啟用 |
| `src/config.py` | `RootCfg.semcom`（預設 None = 不啟用，所有原指令不受影響）|
| `src/main.py` | 建構 semcom 傳入 ModelWrapper |
| `src/model/model_wrapper.py` | `transmit_context_views()`：data_shim 後讓 context views 過通道（target GT 保持乾淨）；記錄傳輸層 `tx_psnr` |
| `src/scripts/train_deepjscc.py` | Stage-1 預訓練（mse / lpips 兩變體）|
| `scripts/run_phase2_sweep.sh` | Phase-2 假設驗證 sweep |
| `scripts/collect_phase2_results.py` | 彙整 sweep 結果成表 |
| `assets/evaluation_index_re10k_smoke.json` | 3 場景迷你評測 index（快速 smoke test 用）|

設計細節：
- Bandwidth ratio R = channel_dim / 96（channel_dim=12 → R=1/8）
- 通道功率 normalize 到 1，AWGN σ² = 10^(−SNR/10)；Rayleigh 為 perfect-CSI 等化等效
- 預訓練以隨機 SNR ∈ [0,20] dB 訓練，單一模型可掃整條 SNR 曲線
- MVSplat checkpoint 不含 semcom 權重，因此啟用 semcom 時 `strict_loading=False`

## Stage-1：JSCC 預訓練

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
python -m src.scripts.train_deepjscc --variant mse   --steps 40000   # 像素保真版
python -m src.scripts.train_deepjscc --variant lpips --steps 40000   # 語意/感知版
# 產出 checkpoints/jscc/jscc_{mse,lpips}_c12.pt（best）與 *_last.pt、log_*.jsonl
```

注意：目前 `datasets/re10k` 只有 41 場景子集（train 5488 frames），JSCC 是小模型尚可，
但結論要上論文前應換完整 RE10K 重訓。

## Phase-2：假設驗證 sweep

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_phase2_sweep.sh
python scripts/collect_phase2_results.py
```

矩陣：4 個 MVSplat 變體（full / base / w/o cost volume / w/o cross-view attn，全部凍結）
× {clean, mse-JSCC, lpips-JSCC} × SNR {0,5,10,15,20}。

判讀方式（看 `d_psnr` = 相對該模型自己 clean 分數的退化量）：
- 若 base 的退化 ≫ wocv 的退化 → cost volume 是脆弱點（假設成立）
- 若 base 的退化 ≫ wobbcrossattn 的退化 → cross-view attention 是脆弱點
- 若 lpips-JSCC 在 tx_psnr 較低但 LPIPS(傳輸層) 較好、而下游退化更大 →
  「保語意、丟空間細節」的取捨確實傷 3D 重建（核心假設）
- 對照 ablation 時用 base（而非 full）當參考，因為 ablation ckpt 都含 wo_depth_refine

## 單次手動評測範例

```bash
python -m src.main +experiment=re10k checkpointing.load=checkpoints/re10k.ckpt \
  mode=test dataset/view_sampler=evaluation test.compute_scores=true \
  +semcom=deepjscc semcom.weights=checkpoints/jscc/jscc_mse_c12.pt semcom.snr_db=10.0
```

快速版加：`dataset.view_sampler.index_path=assets/evaluation_index_re10k_smoke.json`

---

## Phase 3：End-to-End Fine-tune（接線 + 訓練）

新增接線：
- `src/main.py`：semcom 在場時 MVSplat ckpt 用 `strict=False` 載入（保留 JSCC 預訓練權重）。
- `src/model/model_wrapper.py` `configure_optimizers`：param groups（semcom 獨立 LR）；`freeze_mvsplat` 用「只把 semcom 交給 optimizer」實現，**不** set requires_grad=False（否則破壞 MVSplat 內部 autograd 圖，報 "does not require grad"）。
- `OptimizerCfg`：+`semcom_lr`、`freeze_mvsplat`；`TrainerCfg`：+`limit_val_batches`（子集 epoch 太短，設 0 關掉訓練中 validation）。

訓練指令（兩組，detached、各一張 4090、batch 2、20k steps、SNR 隨機、JSCC 從 MSE 預訓練起步）：
```bash
# (b) 只訓 JSCC：optimizer.freeze_mvsplat=true optimizer.semcom_lr=1e-4
# (c) 全 e2e ：optimizer.lr=2e-5 optimizer.semcom_lr=1e-4
python -m src.main +experiment=re10k +semcom=deepjscc \
  checkpointing.load=checkpoints/re10k.ckpt checkpointing.resume=false \
  semcom.weights=checkpoints/jscc/jscc_mse_c12.pt semcom.trainable=true semcom.snr_db=null \
  trainer.max_steps=20000 trainer.limit_val_batches=0 wandb.mode=disabled \
  <上面兩組差異> output_dir=outputs/phase3/<b_jscconly|c_e2e>
```
評測：`bash scripts/run_phase3_eval.sh <ckpt> <tag>`（checkpoint 已含 semcom 權重，semcom.weights 留空）。

## SemCom vs JPEG 比較

JPEG kind 加在 `degradations.py`（容量匹配 bpp = 0.375·log2(1+SNR_lin)，達不到預算則 outage→灰幀=cliff）。
`scripts/run_jpeg_compare.sh` + `scripts/plot_jpeg_compare.py`。

**關鍵結論（誠實）**：在目前 regime（R=1/8、SNR≥0、理想通道碼）JPEG+ideal-coding **贏** DeepJSCC：
- SNR 0→20：JPEG task 23.45→25.95 vs JSCC-MSE 22.30→23.93。
- JPEG 在 SNR≤−5 斷崖（task→11.5），JSCC 理應優雅退化，但需在負 SNR 重訓才看得到其優勢。
- 圖(b)：相同 tx_PSNR 下 JPEG 與 JSCC 落在同一條 task-PSNR 曲線 → 兩者重建「對任務的友善度」相同，差別只在每單位 SNR 能換到多少傳輸品質（理想碼讓 JPEG 換得更多）。
- 這是 JSCC **最不利**的設定。要展現 JSCC 價值需：(a) 負 SNR + JSCC 在該範圍訓練；(b) 實際通道碼（非容量）會在更高 SNR 斷崖；(c) 更低 bandwidth ratio R。
