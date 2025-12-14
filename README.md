# 🚗 Qwen2-VL LoRA 車禍預測模型 - 環境建置指南

本專案使用 **Python 3.10**，並採用 [uv](https://github.com/astral-sh/uv) 進行極速的套件管理與虛擬環境配置。

## 📋 前置需求 (Prerequisites)

  * **OS**: Linux (推薦 Ubuntu 20.04/22.04) 或 Windows (WSL2)
  * **GPU**: NVIDIA 顯卡 (建議 VRAM \>= 24GB)，並已安裝驅動程式。
  * **CUDA**: 建議 CUDA 12.1 或以上版本。

-----

## 🚀 快速開始 (Quick Start)

### 1\. 安裝 uv (如果尚未安裝)

**Linux / macOS:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2\. 初始化專案環境

請依序執行以下指令：

```bash
# 1. 建立 Python 3.10 虛擬環境 (系統會自動下載 managed python 如果沒安裝)
uv venv --python 3.10 .venv

# 2. 啟動虛擬環境
# Linux / macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# 3. 建立 requirements.txt (複製下方內容)
# (請參見下方 "依賴列表" 章節)
```

### 3\. 依賴列表 (requirements.txt)

請在專案根目錄建立 `requirements.txt`，內容如下：

```text
# --- Transformers & LoRA ---
transformers>=4.45.0
accelerate>=0.34.0
huggingface-hub
peft>=0.12.0
bitsandbytes>=0.43.3
qwen-vl-utils>=0.0.8

# --- Utilities ---
numpy<2.0.0
pandas
scikit-learn
pillow
tqdm
tensorboard

# --- Low-level Dependencies ---
pytz
python-dateutil
six
typing_extensions
requests
packaging
pyyaml
```

### 4\. 安裝套件

使用 `uv` 進行極速安裝：

```bash
# 步驟 A: 安裝 PyTorch (指定官方 CU121 倉庫)
uv pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# 步驟 B: 安裝主要依賴
uv pip install -r requirements.txt

# 步驟 C: 安裝 Flash Attention 2 (強烈建議，加速訓練與節省顯存)
# 注意：這一步需要系統有安裝 CUDA Toolkit (nvcc)
uv pip install flash-attn --no-build-isolation
```

> **⚠️ Flash Attention 安裝失敗怎麼辦？**
> 如果步驟 B 報錯，通常是因為編譯環境問題。你可以嘗試下載預編譯好的 wheel 檔安裝 (以 Linux, Python 3.10, Torch 2.4, CUDA 12.x 為例)：
>
> ```bash
> uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation
> ```

-----

## ✅ 驗證環境

建立一個 `check_env.py` 檔案並執行，確認環境是否就緒：

```python
import torch
import sys
try:
    import flash_attn
    fa_status = "Installed ✅"
except ImportError:
    fa_status = "Not Found ⚠️ (Training will be slower)"

print(f"=======================================")
print(f"Python Version: {sys.version.split()[0]}")
print(f"Torch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"BF16 Support: {torch.cuda.is_bf16_supported()}")
print(f"Flash Attention: {fa_status}")
print(f"=======================================")
```

執行檢查：

```bash
python check_env.py
```

-----

## 📂 建議的專案結構

為了配合訓練程式碼，建議你的專案目錄結構如下：

```text
project_root/
├── .venv/                 # uv 建立的虛擬環境
├── requirements.txt       # 套件清單
├── README.md              # 本說明檔
├── train_lora.py          # 主要訓練程式碼
├── output.csv             # 訓練資料索引 (Train)
├── gt_public.csv          # 驗證資料索引 (Val)
└── dataset/               # (建議) 透過 Symbolic Link 連結到本目錄
```