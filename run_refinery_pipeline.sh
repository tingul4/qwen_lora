#!/bin/bash
set -e

# Activate uv venv
# source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. Train Seed LoRA
echo "Starting Seed LoRA Training..."
CUDA_VISIBLE_DEVICES=0 python3 train_seed_lora.py \
    --dataset_root /raid/mystery-project/dataset \
    --csv_path /raid/mystery-project/dataset/road_train_and_val_with_reports_revised.csv \
    --epochs 10 \
    --output_dir checkpoints_seed_lora \
    --num_frames 30

# Get the best model path (assuming it's saved in logs/seed_TIMESTAMP/best_seed_model)
# We need to find the latest log dir
LATEST_LOG_DIR=$(ls -td logs/seed_* | head -1)
BEST_MODEL_PATH="${LATEST_LOG_DIR}/best_seed_model"

echo "Best Seed Model found at: ${BEST_MODEL_PATH}"

# 2. Generate Pseudo Labels
echo "Generating Pseudo Labels..."
CUDA_VISIBLE_DEVICES=0 python3 generate_pseudo_labels_refined.py \
    --lora_path "${BEST_MODEL_PATH}" \
    --csv_path /raid/mystery-project/dataset/road_train_and_val_with_reports_revised.csv \
    --output_csv /raid/mystery-project/dataset/road_train_and_val_pseudo.csv

echo "Pipeline Completed. Final dataset is at /raid/mystery-project/dataset/road_train_and_val_pseudo.csv"
