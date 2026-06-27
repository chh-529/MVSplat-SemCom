# MVSplat + SemCom 實驗筆記（Part 1：架構、SemCom、資料集、訓練）

> 系統概念：把 MVSplat（feed-forward 3D Gaussian Splatting）當作接收端的下游任務。
> 發送端用 SemCom（DeepJSCC）把 context views 經過雜訊通道送到接收端，接收端重建 3D 場景並渲染新視角。
> 核心問題：通道退化如何影響 3D 重建？SemCom 相對傳統「壓縮 + 通道碼」有何優劣？

---

## 1. 兩種架構（SemCom 插入點）

MVSplat encoder 內部資料流：
```
影像 → per-view CNN → cnn_features → cross-view Transformer → cost volume → Gaussians → 渲染
       [逐 view 獨立]   [128×64×64]   [需所有 view 一起]
```
cross-view attention 之後必須在接收端（需要所有 view），所以 SemCom 只能插在它之前。

### 架構 A — Pixel domain（主軸）
傳**影像**。發送端只送 context views，接收端跑**完整** MVSplat。
```
[發送端] 影像 ──JSCC enc→通道→JSCC dec──→ [接收端] 影像_hat → 完整 MVSplat → 渲染
```
- 實作：`model_wrapper.transmit_context_views`（在 encoder 之前套用）
- 優點：發送端計算輕、傳的是通用影像
- 對應傳統 baseline：**JPEG + 理想通道碼**

### 架構 B — Feature domain（插入點 B）
傳 **CNN 特徵**。發送端跑 per-view CNN，接收端從特徵繼續。
```
[發送端] 影像→CNN→特徵 ──JSCC enc→通道→JSCC dec──→ [接收端] 特徵_hat → Transformer → cost volume → 渲染
```
- 實作：`backbone_multiview.forward` 在 CNN 後、Transformer 前套用 `feature_semcom`
- 特性：task-oriented（直接傳任務相關特徵）；發送端需跑 CNN、且收發兩端要共享 CNN 權重
- 對應傳統 baseline：**PCA + 純量量化 + 理想通道碼**（feature 版 transform coding）

---

## 2. SemCom 詳細說明（DeepJSCC）

**類比式聯合信源-通道編碼（analog joint source-channel coding）**：把來源直接映射成連續數值的 channel symbols 送出，**不經過 bits / 量化**（與數位的 JPEG 相反）。

### 通道模型（兩種架構共用）
1. **Power normalization**：把 latent 正規化到單位平均功率
2. **AWGN 通道**：`y = z + n`，`n ~ N(0, σ²)`，`σ² = 10^(−SNR/10)`
3. （可選 Rayleigh fading，perfect-CSI 等化）

### 頻寬 / 壓縮率
- latent = `channel_dim × 64 × 64`，channel_dim = **12**
- 複數 channel uses `k = 12×64×64/2 = 24,576`／view
- **Bandwidth ratio R = k / (3·H·W) = 1/8**（每 8 個 source 樣本用 1 個複數符號）
- 兩種架構用**同一個 12×64×64 瓶頸 → 同 R=1/8，可公平比較**

### Encoder/Decoder 結構
| | Pixel-JSCC（架構 A）| Feature-JSCC（架構 B）|
|---|---|---|
| 輸入 | 3×256×256 影像 | 128×64×64 特徵 |
| Encoder | 5 層 Conv+PReLU，2 個 stride-2（降採樣 ×4）：3→16→32→32→32→**12** | 3 層 Conv+PReLU（不降採樣）：128→64→32→**12** |
| 瓶頸 | 12×64×64 | 12×64×64 |
| Decoder | 鏡像反卷積，末層 Sigmoid（輸出 [0,1]）| 鏡像，末層無 activation（特徵非 [0,1]）|
| 壓縮倍率 | 4× | 10.7× |
| 參數量 | ~150 K | ~192 K |

---

## 3. 傳統壓縮 baseline（對照組）

| 領域 | 傳統方法 | 說明 |
|---|---|---|
| Pixel（架構 A）| **JPEG + 理想通道碼** | 影像→DCT→量化→bits；理想碼下 bit 預算 `B = k·log₂(1+SNR)`，bpp = 0.375·log₂(1+SNR)；塞不下→outage（斷崖）|
| Feature（架構 B）| **PCA + 純量量化 + 理想通道碼** | 特徵→PCA(128→top-P)→2-bit 量化；P 由 bit 預算決定（SNR0→3、SNR10→10、SNR20→19 個主成分）；P<1→outage |

「理想通道碼」= 假設達到 Shannon 容量的糾錯碼（容量以下零錯、以上失敗）。這給傳統系統**最佳情況**（steel-man）。

---

## 4. 資料集

**RealEstate10K (RE10K)**：YouTube 室內/房地產影片，640×360 frames + SfM 相機姿態（無 GT 深度）。

| 資料集 | train | test | 用途 |
|---|---|---|---|
| `datasets/re10k`（原始小子集）| 39 場景 | 41 場景 | 早期 Phase 1–3（結論受資料量限制）|
| `datasets/re10k_big`（自建）| **3,000 場景** | **500 場景**（約 429 實際可評測*）| 正式實驗 |

\* 其餘約 71 個被 MVSplat 的 FOV / view-sampler 條件跳過。

**自建流程（Route C）**：`scripts/download_re10k.py`——官方 pose tarball（train 71,556 / test 7,711）→ yt-dlp 抓 360p → 單遍抽指定 timestamp 的 frame → 打包成 pixelSplat chunk 格式。已端到端驗證（下載場景跑 MVSplat 得 PSNR 26.2，與原資料同級）。test 場景過濾成 `evaluation_index_re10k.json` 內的 key 才可評測。

**評測切分**：mode=test 用 `datasets/*/test`，每場景 2 個 context views（由 `evaluation_index_re10k.json` 指定）。train/test 場景**完全不重疊**（無洩漏）。

---

## 5. 訓練細節

**接收端 MVSplat**：官方預訓練 `re10k.ckpt`（完整 RE10K、300k steps、單張 A100）。

### Stage 1 — Pixel-JSCC 預訓練（影像傳輸，MVSplat 不參與）
| 項目 | 設定 |
|---|---|
| 變體 | **MSE**（loss=MSE，像素保真）／**LPIPS**（loss=0.1·MSE+LPIPS，感知/語意）|
| 步數 | 40,000 |
| Optimizer | Adam，lr **1e-4**，CosineAnnealing |
| Batch | 32（256×256 random crop）|
| SNR | 隨機均勻抽（早期 [0,20]，後期 [−10,20]）|
| 資料 | RE10K frames（`max_frames` 上限 200k 控 RAM）|
| 收斂 | MSE val PSNR@10dB ≈ 28.4；LPIPS ≈ 25.1（都打平）|

### Stage 2 — End-to-End Fine-tune
從 `re10k.ckpt` + 預訓練 JSCC 出發，rendering loss 反傳穿過 AWGN 通道（可微）。
| 項目 | 設定 |
|---|---|
| 步數 | 20,000 |
| Batch | 2／GPU（4090 24GB）|
| LR（param groups）| MVSplat **2e-5**、JSCC **1e-4**（OneCycleLR cosine）|
| SNR | 訓練時每樣本隨機抽 [−10,20]（評測時固定、掃多個 run）|
| Rendering loss | **MSE 權重 1.0 + LPIPS 權重 0.05**（apply_after_step=0）|
| 三種設定 | (a) 全凍結 baseline／(b) 只訓 JSCC（freeze_mvsplat）／(c) 全 e2e |

### Feature-JSCC 訓練（架構 B）
特徵無自然的重建目標 → **直接用 rendering loss 訓練**（task-oriented），不做 Stage-1 預訓練。其餘設定同 Stage 3（20k steps、batch 2、SNR [−10,20]、(b)/(c) 兩種）。

### 通道內 SNR 處理
- **訓練**：`snr_db=null` → 每個樣本獨立隨機抽 SNR（一個 batch 內不同場景可不同 SNR）→ 一組權重對整個範圍 robust
- **評測**：`snr_db=固定值` → 該 run 所有場景同 SNR；掃多個 run 畫曲線
- 註：目前是 **SNR-randomized**（網路看不到 SNR），**非** adaptive-SNR（ADJSCC 那種把 SNR 當條件輸入）

### 硬體 / 環境
2× RTX 4090（gpu3），conda env `mvsplat`，torch 2.5.1+cu121，diff_gaussian_rasterization 已編譯。
