# MVSplat 環境搭建 / 搬機指南

記錄這份 project 在 gpu4 上跑起來的完整流程，以及搬到別台機器（如 gpu3）要注意的事。
最後可用狀態：2026-06-11 於 gpu4，conda env `mvsplat`，RE10K 子集評測 PSNR 26.02。

---

## 0. 重要前提：哪些東西是「本機」的

| 路徑 | 位置 | 搬機時 |
|---|---|---|
| `~/miniconda3`（含 env `mvsplat`） | gpu4 **本機磁碟** `/` | gpu3 看不到，要搬 |
| `/tmp2/b12902145/mvsplat`（本專案） | gpu4 **本機 nvme** | gpu3 看不到，要搬 |
| `/rhome/...`（NFS home） | 共享 | 兩台都看得到，可當中轉 |

→ **環境和專案都得搬**。下面 Path A 連環境一起打包（不必重編譯），Path B 在 gpu3 重建。

---

## 1. 踩過的坑（無論哪條路都要知道）

1. **torch 版本跟 README 不同**：實際用的是 `torch==2.5.1+cu121`，不是 README 寫的 2.1.2+cu118。
2. **三個套件要降版**，否則 import 失敗：
   - `numpy==1.26.4`（opencv 4.6 不相容 numpy 2.x）
   - `moviepy==1.0.3`（2.x 移除了 `moviepy.editor`）
   - `plyfile==1.0.3`
3. **`diff_gaussian_rasterization` 需編譯**：用 env 自帶的 `nvcc 12.1`（在 `~/miniconda3/envs/mvsplat/bin/nvcc`），**不要**用系統的 CUDA 13.2，版本對不上。
4. **GPU 編號陷阱**：gpu4 有一張 RTX 5090（sm_120），torch cu121 不支援它，且 CUDA 的裝置序號跟 `nvidia-smi` 不一致。跑任何指令前先設：
   ```bash
   export CUDA_DEVICE_ORDER=PCI_BUS_ID
   export CUDA_VISIBLE_DEVICES=2   # 指到支援的 GPU（在 gpu4 是閒置的 4090）
   ```
   **搬到 gpu3 後，先用 `nvidia-smi` 確認那台的 GPU 型號**：若都是 sm_90 以下（A100/3090/4090 等）就沒問題；若也有 50 系列就得換 torch（見最後一節）。

---

## Path A（推薦）：用 conda-pack 整包搬，不重編譯

最省事、最不會再踩坑，因為連編好的 rasterizer 一起搬。前提：兩台 OS/CPU 架構相同（同實驗室機器通常都是 Ubuntu 24.04 x86_64）。

### 在 gpu4（來源機）
```bash
conda activate base
pip install conda-pack            # base 還沒裝
cd /tmp2/b12902145/mvsplat

# 1) 打包 conda 環境（產出約數 GB 的 tar）
conda pack -n mvsplat -o /rhome/$USER/mvsplat_env.tar.gz   # 放 NFS 當中轉

# 2) 打包專案本身（程式碼 + checkpoints + 資料子集）
#    若 gpu3 的 /tmp2 空間夠，整包一起搬；資料大可分開傳
tar czf /rhome/$USER/mvsplat_proj.tar.gz \
    --exclude='outputs' --exclude='.git' \
    -C /tmp2/b12902145 mvsplat
```

### 在 gpu3（目標機）
```bash
# 1) 解環境（放回自己的 conda envs 目錄）
mkdir -p ~/miniconda3/envs/mvsplat
tar xzf /rhome/$USER/mvsplat_env.tar.gz -C ~/miniconda3/envs/mvsplat
conda activate mvsplat
conda-unpack            # 修正環境內寫死的路徑，必跑

# 2) 解專案
mkdir -p /tmp2/$USER && tar xzf /rhome/$USER/mvsplat_proj.tar.gz -C /tmp2/$USER
cd /tmp2/$USER/mvsplat

# 3) 驗證
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0     # 依 gpu3 實際情況挑
python -c "import torch, diff_gaussian_rasterization; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> 注意：NFS home 可能有容量限制，環境 tar 若太大就改用 `scp` 直接機器對傳，或暫存到 gpu3 的 `/tmp2`。

---

## Path B（備援）：在 gpu3 重新建環境

若 conda-pack 失敗（例如兩台 glibc/CUDA driver 差異太大），就照原本流程重建。
rasterizer 會重新編譯，需要 env 內的 nvcc，所以照下面順序裝。

```bash
conda create -n mvsplat python=3.10 -y
conda activate mvsplat

# 1) 先裝 torch（決定 CUDA ABI），rasterizer 要對齊它
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 2) 裝 nvcc 12.1（編 rasterizer 用，跟 torch 的 cu121 對齊）
conda install -c nvidia cuda-nvcc=12.1 -y
# 或 pip install nvidia-cuda-nvcc-cu12==12.1.*

# 3) 其餘套件（直接用凍結檔最保險）
cd /tmp2/$USER/mvsplat
pip install -r requirements_frozen.txt

# 若不用 frozen，改用原始 requirements 並手動修版本：
# pip install -r requirements.txt
# pip install numpy==1.26.4 moviepy==1.0.3 plyfile==1.0.3
# pip install git+https://github.com/dcharatan/diff-gaussian-rasterization-modified
```

`requirements_frozen.txt` 是 gpu4 上 `pip freeze` 的完整快照（含 rasterizer 的 git commit），是最可靠的還原依據。

---

## 2. 搬完後的冒煙測試（RE10K 子集評測）

```bash
cd /tmp2/$USER/mvsplat
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=<挑一張支援的 GPU>
python -m src.main +experiment=re10k \
  checkpointing.load=checkpoints/re10k.ckpt \
  mode=test dataset/view_sampler=evaluation \
  test.compute_scores=true
```
預期：PSNR ≈ 26.0 / SSIM ≈ 0.87 / LPIPS ≈ 0.12（41 場景子集）。對得上就搬成功。

---

## 3. 萬一 gpu3 也有 50 系列 GPU（sm_120）

torch cu121 不支援 sm_120，會報 `no kernel image is available`。兩個選擇：
- 用 `CUDA_VISIBLE_DEVICES` 避開那張卡，挑舊卡跑（最簡單）。
- 若只剩 50 系列可用，得換成支援 sm_120 的 torch（cu128 nightly 或更新版），
  此時 `diff_gaussian_rasterization` 必須對新 torch **重新編譯**（走 Path B 流程，nvcc 也要換對應版本）。
