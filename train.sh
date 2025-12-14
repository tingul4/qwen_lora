CUDA_VISIBLE_DEVICES=3 python train_lora.py \
    --epochs 30 \
    --dataset_root /raid/mystery-project/dataset \
    --freeway_train_csv /raid/mystery-project/dataset/freeway_train_with_reports.csv \
    --road_train_csv /raid/mystery-project/dataset/road_train_with_reports.csv \
    --val_csv /raid/mystery-project/qwen_lora/gt_public.csv