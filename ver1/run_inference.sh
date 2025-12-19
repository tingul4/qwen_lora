#!/bin/bash

# Set CUDA device if needed
export CUDA_VISIBLE_DEVICES=1

# Run the inference script
/raid/mystery-project/qwen_lora/.venv/bin/python /raid/mystery-project/qwen_lora/sam3_qwen_pipeline/inference.py
