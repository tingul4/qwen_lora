# 請將 --adapter_path 指向第一步訓練出來的最佳 checkpoint 資料夾
CUDA_VISIBLE_DEVICES=0 python generate_pseudo_labels.py \
    --dataset_root /raid/mystery-project/dataset \
    --road_train_csv /raid/mystery-project/dataset/road_train_and_val_with_reports_revised.csv \
    --adapter_path /raid/mystery-project/qwen_lora/logs/12160006/best_model_auc_0.5644_f1_0.6667 \
    --output_dir refined_labels