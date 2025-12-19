
import transformers
print(f"Transformers version: {transformers.__version__}")
try:
    from transformers import AutoModelForDepthEstimation
    print("AutoModelForDepthEstimation available")
except ImportError:
    print("AutoModelForDepthEstimation NOT available")
