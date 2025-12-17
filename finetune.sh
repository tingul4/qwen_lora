CUDA_VISIBLE_DEVICES=0 python train_lora.py \
    --epochs 5 \
    --dataset_root /raid/mystery-project/dataset \
    --freeway_train_csv /raid/mystery-project/dataset/freeway_train_with_reports.csv \
    --road_train_csv /raid/mystery-project/qwen_lora/splits/train_split.csv \
    --val_csv /raid/mystery-project/qwen_lora/splits/val_split.csv \
    --output_dir checkpoints_initial_stage \
    --only_use_labeled_data