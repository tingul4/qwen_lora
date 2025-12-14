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