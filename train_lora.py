import os
import csv
import glob
import random
import logging
import argparse
import numpy as np
import pandas as pd
import torch
from datetime import datetime
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score, precision_recall_curve
from PIL import Image
from qwen_vl_utils import process_vision_info

from transformers import (
    Qwen2VLForConditionalGeneration, 
    AutoProcessor, 
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType, 
    prepare_model_for_kbit_training
)

# ===================== Logging Setup =====================
# We will configure logging inside main() to use the timestamped directory
LOGGER = logging.getLogger("train")

# ===================== Utils =====================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_frames_uniformly(folder_path, num_frames=8):
    """
    從資料夾中採樣圖片
    如果 num_frames == -1，採樣所有圖片
    否則均勻採樣 num_frames 張圖片
    """
    exts = ["*.jpg", "*.png", "*.jpeg", "*.JPG"]
    all_files = []
    for e in exts:
        all_files.extend(glob.glob(os.path.join(folder_path, e)))
    all_files = sorted(all_files)
    
    total_frames = len(all_files)
    if total_frames == 0:
        return []
    
    # 如果 num_frames == -1，採樣所有圖片
    if num_frames == -1:
        return all_files
    
    if total_frames < num_frames:
        return [all_files[i % total_frames] for i in range(num_frames)]
    
    indices = np.linspace(0, total_frames - 1, num_frames + 1, dtype=int)
    selected_paths = []
    for i in range(num_frames):
        idx = (indices[i] + indices[i+1]) // 2 
        selected_paths.append(all_files[idx])
        
    return selected_paths

def load_train_data(dataset_root, freeway_csv, road_csv, only_use_labeled_data=False):
    """
    讀取訓練資料：明確指定 Freeway 與 Road 的 CSV 位置
    """
    all_data = []
    
    # 1. 處理 Freeway Train
    # 路徑假設: dataset_root/freeway/train/FILE_NAME
    if os.path.exists(freeway_csv):
        df = pd.read_csv(freeway_csv)
        LOGGER.info(f"Loading Freeway Train from: {freeway_csv} ({len(df)} rows)")
        for _, row in df.iterrows():
            fname = str(row["file_name"]).strip()
            # 組合路徑
            full_path = os.path.join(dataset_root, "freeway", "train", fname)
            
            report = ""
            if "report" in row and not pd.isna(row["report"]):
                report = str(row["report"]).strip()
            
            if only_use_labeled_data and not report:
                continue

            all_data.append({
                "folder_path": full_path,
                "file_name": fname,
                "risk": float(row["risk"]),
                "report": report,
                "type": "freeway_train"
            })
    else:
        LOGGER.warning(f"Freeway CSV not found: {freeway_csv}")

    # 2. 處理 Road Train
    # 路徑假設: dataset_root/road/train/FILE_NAME
    if os.path.exists(road_csv):
        df = pd.read_csv(road_csv)
        LOGGER.info(f"Loading Road Train from: {road_csv} ({len(df)} rows)")
        for _, row in df.iterrows():
            fname = str(row["file_name"]).strip()
            # 組合路徑
            full_path = os.path.join(dataset_root, "road", "train", fname)
            
            report = ""
            if "report" in row and not pd.isna(row["report"]):
                report = str(row["report"]).strip()
            
            if only_use_labeled_data and not report:
                continue

            all_data.append({
                "folder_path": full_path,
                "file_name": fname,
                "risk": float(row["risk"]),
                "report": report,
                "type": "road_train"
            })
    else:
        LOGGER.warning(f"Road CSV not found: {road_csv}")

    LOGGER.info(f"Total Training Samples: {len(all_data)}")
    if len(all_data) == 0:
        raise ValueError("No training data found.")
        
    return all_data

def load_val_data(dataset_root, val_csv):
    """
    讀取驗證資料 (gt_public.csv)
    自動在 freeway/test 和 road/test 尋找對應資料夾
    """
    if not os.path.exists(val_csv):
        raise ValueError(f"Validation CSV not found: {val_csv}")
        
    df = pd.read_csv(val_csv)
    LOGGER.info(f"Loading Validation from: {val_csv} ({len(df)} rows)")
    
    val_data = []
    found_count = 0
    missing_count = 0
    
    # 預先定義可能的搜尋路徑
    search_dirs = [
        os.path.join(dataset_root, "freeway", "test"),
        os.path.join(dataset_root, "road", "test"),
        os.path.join(dataset_root, "freeway", "train"),
        os.path.join(dataset_root, "road", "train")
    ]
    
    for _, row in df.iterrows():
        fname = str(row["file_name"]).strip()
        risk = float(row["risk"])
        
        target_path = None
        # 搜尋檔案存在於哪個資料夾
        for d in search_dirs:
            candidate = os.path.join(d, fname)
            if os.path.isdir(candidate):
                target_path = candidate
                break
        
        if target_path:
            val_data.append({
                "folder_path": target_path,
                "file_name": fname,
                "risk": risk,
                "type": "val"
            })
            found_count += 1
        else:
            # 為了避免驗證時報錯，如果找不到實體檔案則跳過
            missing_count += 1

    LOGGER.info(f"Val Samples Found: {found_count} | Missing: {missing_count}")
    return val_data

# ===================== Dataset =====================
class AccidentDataset(Dataset):
    def __init__(self, data_list, processor, num_frames=8, mode="train"):
        self.data = data_list
        self.processor = processor
        self.num_frames = num_frames
        self.mode = mode
        
        self.user_prompt = "Is there an accident or potential danger in this video? Answer only with 'Yes' or 'No'."
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        folder_path = item["folder_path"]
        risk = item["risk"]
        
        answer = "Yes" if risk >= 0.5 else "No"
        
        # 1. 讀取圖片 (PIL)
        frame_paths = get_frames_uniformly(folder_path, num_frames=self.num_frames)
        if len(frame_paths) == 0:
            return self.__getitem__(random.randint(0, len(self.data) - 1))

        frames = []
        for p in frame_paths:
            try:
                img = Image.open(p).convert("RGB")
                frames.append(img)
            except Exception:
                pass
        
        if len(frames) == 0:
             return self.__getitem__(random.randint(0, len(self.data) - 1))

        # [修改 1] 處理 Report
        # 讀取 report，如果沒有或是 NaN，給一個預設值
        report = item.get("report", "")
        if pd.isna(report) or report == "":
            if risk >= 0.5:
                report = "Detected potential traffic conflict in the scene."
            else:
                report = "Traffic flow appears normal and safe."
        
        # [修改 2] 定義 CoT 格式
        # 格式： "Analysis: {report}\nConclusion: {answer}"
        # 這樣模型才會學到要先講 Analysis
        if risk >= 0.5:
            answer_content = f"Analysis: {report}\nConclusion: Yes"
        else:
            answer_content = f"Analysis: {report}\nConclusion: No"
        
        # 2. 準備原始 Messages (給 process_vision_info 用)
        messages_raw = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames}, 
                    {"type": "text", "text": self.user_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]
        
        # 3. 提取視覺特徵 (這是最重要的步驟，取得 pixel values)
        # process_vision_info 會幫我們把 PIL 列表轉成 Tensor
        image_inputs, video_inputs = process_vision_info(messages_raw)
        
        # 4. 【核彈級修正】準備文字生成的 Messages
        # 我們不讓 apply_chat_template 去猜 video 是什麼，我們直接塞入 Qwen2-VL 指定的佔位符
        # 佔位符格式: <|vision_start|><|video_pad|><|vision_end|>
        
        video_token_str = "<|vision_start|><|video_pad|><|vision_end|>"
        
        messages_for_text = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": video_token_str}, 
                    {"type": "text", "text": self.user_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": answer_content # 放入 CoT 內容
            }
        ]

        # 5. 生成 Prompt String
        text = self.processor.apply_chat_template(
            messages_for_text, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        # 6. Processor 打包
        if self.mode == "train":
            inputs = self.processor(
                text=[text],
                videos=video_inputs, # 這裡傳入從 messages_raw 提取出的 video tensors
                padding=True,
                return_tensors="pt",
            )
            # [ADD] 加入這行，讓訓練迴圈知道 Ground Truth
            inputs["risk"] = risk
            # [重要] 讓 input_ids 當作 labels，這樣模型才會計算生成文字的 Loss
            inputs["labels"] = inputs["input_ids"].clone()
            # Mask padding tokens
            if "attention_mask" in inputs:
                inputs["labels"][inputs["attention_mask"] == 0] = -100
        else:
            # 驗證模式：去掉 answer
            messages_val = messages_for_text[:-1]
            text_val = self.processor.apply_chat_template(messages_val, tokenize=False, add_generation_prompt=True)
            
            # 驗證時也要傳入 video inputs
            inputs = self.processor(
                text=[text_val],
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs["ground_truth_risk"] = risk
            inputs["file_name"] = item["file_name"]

        # 7. 鍵值修正 (Key Remapping)
        # We do NOT want to rename pixel_values_videos to pixel_values for Qwen2-VL
        # because the model distinguishes between image (pixel_values) and video (pixel_values_videos).
        # if "pixel_values_videos" in inputs:
        #     inputs["pixel_values"] = inputs.pop("pixel_values_videos")
        
        # if "video_grid_thw" in inputs:
        #     inputs["image_grid_thw"] = inputs.pop("video_grid_thw")

        # 8. Squeeze Batch Dim
        for k, v in inputs.items():
            if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == 1:
                inputs[k] = v.squeeze(0)
                
        # 最終防呆
        if "pixel_values_videos" not in inputs:
            return self.__getitem__(random.randint(0, len(self.data) - 1))

        return inputs

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return batch 

# ===================== Evaluation =====================
@torch.no_grad()
def evaluate_auc(model, dl_val, processor, device):
    model.eval()
    all_probs = []
    all_labels = []
    
    # 取得 Token IDs
    yes_ids = processor.tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = processor.tokenizer.encode("No", add_special_tokens=False)
    yes_token_id = yes_ids[0]
    no_token_id = no_ids[0]

    # 使用 tqdm 顯示進度
    for batch in tqdm(dl_val, desc="Evaluating", leave=False):
        for sample in batch:
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
            
            pixel_values_videos = sample["pixel_values_videos"].to(device)
            video_grid_thw = sample["video_grid_thw"].unsqueeze(0).to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw
            )
            
            logits = outputs.logits[0, -1, :]
            score_yes = logits[yes_token_id].item()
            score_no = logits[no_token_id].item()
            
            prob_yes = np.exp(score_yes) / (np.exp(score_yes) + np.exp(score_no) + 1e-6)
            
            all_probs.append(prob_yes)
            all_labels.append(1 if sample["ground_truth_risk"] >= 0.5 else 0)

    if not all_labels: 
        return 0.0, 0.0, 0.0 # Return auc, acc, f1

    try:
        auc = roc_auc_score(all_labels, all_probs)
        
        # [修改] 尋找最佳門檻值 (Best Threshold)
        precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
        # F1 = 2 * (P * R) / (P + R)
        # 注意: thresholds 的長度比 precisions/recalls 少 1 (最後一個 precision=1, recall=0 沒有對應 threshold)
        # 但實務上可以直接忽略最後一點或處理長度問題
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
        
        # 找出最大 F1 的索引 (最後一個點通常是 0，忽略)
        # f1_scores[:-1] 對應 thresholds
        best_idx = np.argmax(f1_scores[:-1]) if len(f1_scores) > 1 else 0
        best_threshold = thresholds[best_idx] if len(thresholds) > best_idx else 0.5
        best_f1 = f1_scores[best_idx]
        
        LOGGER.info(f"Optimal Threshold Found: {best_threshold:.4f} (Max F1: {best_f1:.4f})")
        
        # 使用最佳門檻值進行預測
        preds = [1 if p >= best_threshold else 0 for p in all_probs]
        acc = accuracy_score(all_labels, preds)
        
        # 雖然上面算過 best_f1，這裡用標準函式再確認一次
        f1 = f1_score(all_labels, preds)
        
        tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
        LOGGER.info(f"Confusion Matrix (Thresh={best_threshold:.3f}): TN={tn} FP={fp} FN={fn} TP={tp}")
        
    except ValueError:
        auc = 0.5
        acc = 0.0
        f1 = 0.0
        
    return auc, acc, f1

# ===================== Main =====================
def main():
    parser = argparse.ArgumentParser()
    
    # 路徑設定
    parser.add_argument("--dataset_root", type=str, default="/raid/mystery-project/dataset")
    parser.add_argument("--freeway_train_csv", type=str, default="/raid/mystery-project/dataset/freeway_train.csv")
    parser.add_argument("--road_train_csv", type=str, default="/raid/mystery-project/dataset/road_train_and_val_with_reports_revised.csv")
    parser.add_argument("--val_csv", type=str, default="/raid/mystery-project/qwen_lora/gt_public.csv")
    
    # 訓練參數
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=8, help="Number of frames to sample from each video. Use -1 to sample all frames.")
    parser.add_argument("--output_dir", type=str, default="checkpoints_lora")
    parser.add_argument("--only_use_labeled_data", action="store_true", help="Only use data with existing reports for training")
    args = parser.parse_args()

    set_seed(42)
    
    # ===================== New Logging Setup =====================
    timestamp = datetime.now().strftime("%m%d%H%M")
    log_dir = os.path.join("logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    # Re-configure logging to write to file and console
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "train.log")),
            logging.StreamHandler()
        ]
    )
    
    # Update global LOGGER
    global LOGGER
    LOGGER = logging.getLogger("train")
    LOGGER.info(f"Logging to: {log_dir}")

    # TensorBoard Writer
    writer = SummaryWriter(log_dir=log_dir)

    # 1. 準備資料
    LOGGER.info("Step 1: Loading Training Data...")
    train_data = load_train_data(
        args.dataset_root, 
        args.freeway_train_csv, 
        args.road_train_csv,
        only_use_labeled_data=args.only_use_labeled_data
    )
    
    # Log frame sampling configuration
    if args.num_frames == -1:
        LOGGER.info(f"Frame Sampling: Sample ALL frames from each video")
    else:
        LOGGER.info(f"Frame Sampling: Uniformly sample {args.num_frames} frames from each video")
    
    # ===================== Class Balancing (Weighted Sampler) =====================
    # Count Pos/Neg
    pos_count = sum(1 for x in train_data if x["risk"] >= 0.5)
    neg_count = len(train_data) - pos_count
    LOGGER.info(f"Class Distribution | Positive (Yes): {pos_count} | Negative (No): {neg_count}")
    
    # 權重設為 Negative/Positive 的比例，這裡大約是 2.1
    # [修改] 強制設定為 5.0 以懲罰 False Negatives
    # pos_weight_val = neg_count / max(1, pos_count) 
    pos_weight_val = 5.0
    LOGGER.info(f"Forcing pos_weight to: {pos_weight_val} (Original calc was {neg_count/max(1, pos_count):.2f})")
    
    # Calculate Weights
    # Prevent division by zero
    weight_pos = 1.0 / pos_count if pos_count > 0 else 0.0
    weight_neg = 1.0 / neg_count if neg_count > 0 else 0.0
    
    # samples_weights = []
    # for x in train_data:
    #     if x["risk"] >= 0.5:
    #         samples_weights.append(weight_pos)
    #     else:
    #         samples_weights.append(weight_neg)
            
    # samples_weights = torch.tensor(samples_weights, dtype=torch.double)
    # sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
    # ==============================================================================
    
    LOGGER.info("Step 2: Loading Validation Data...")
    val_data = load_val_data(
        args.dataset_root,
        args.val_csv
    )

    # 2. 載入模型
    LOGGER.info("Step 3: Loading Qwen2-VL Model (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    # 3. LoRA 設定
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. DataLoader
    train_ds = AccidentDataset(train_data, processor, num_frames=args.num_frames, mode="train")
    val_ds = AccidentDataset(val_data, processor, num_frames=args.num_frames, mode="val")
    
    # Use sampler instead of shuffle=True
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # 5. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_dl) * args.epochs // args.grad_accum
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)

    # 6. Training Loop
    global_step = 0
    best_auc = 0.0 # Keep best_auc for saving the model
    
    # pos_weight 必須是一個 Tensor
    loss_fct = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(model.device))
    
    # 取得 "Yes" 和 "No" 的 Token ID (為了拿 Logits)
    yes_ids = processor.tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = processor.tokenizer.encode("No", add_special_tokens=False)
    yes_token_id = yes_ids[0]
    no_token_id = no_ids[0]
    
    # 定義分類權重 Loss (針對 Yes/No)
    cls_loss_fct = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(model.device))

    LOGGER.info("Starting Training with Hybrid CoT Loss...")
    for epoch in range(args.epochs):
        model.train()
        
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            batch_loss = 0.0
            optimizer.zero_grad()
            
            for sample in batch:
                # 1. input_ids 和 attention_mask 已經有 unsqueeze(0) 了 (正確)
                input_ids = sample["input_ids"].unsqueeze(0).to(model.device)
                attention_mask = sample["attention_mask"].unsqueeze(0).to(model.device)
                
                # 2. pixel_values_videos: 
                pixel_values_videos = sample["pixel_values_videos"].to(model.device)
                
                # 3. video_grid_thw
                video_grid_thw = sample["video_grid_thw"].unsqueeze(0).to(model.device)
                
                # [修改 3] Forward Pass
                # 傳入 labels，讓模型自動計算 LM Loss (這會包含 report 的生成學習)
                labels = sample["labels"].unsqueeze(0).to(model.device)
                
                # [MODIFY] Forward 改成不傳 labels (這樣模型不會自己算 loss，節省一點點資源)
                # 或者傳了也沒關係，我們不用 outputs.loss
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values_videos=pixel_values_videos,
                    video_grid_thw=video_grid_thw,
                    labels=labels,
                    use_cache=False 
                )
                
                # Loss 1: Causal LM Loss (語言生成損失)
                # 這是讓模型學會寫 report 的關鍵
                lm_loss = outputs.loss 
                
                # Loss 2: Weighted Classification Loss (分類加權損失)
                # 這是解決資料不平衡的關鍵
                # 我們依然去抓最後一個 token 的 logits 來強化 Yes/No 的判斷
                logits = outputs.logits[:, -1, :] 
                score_yes = logits[:, yes_token_id]
                score_no = logits[:, no_token_id]
                diff_logits = score_yes - score_no
                
                # 準備 Target (因為 batch=1)
                current_risk = sample["risk"]
                target = torch.tensor([1.0 if current_risk >= 0.5 else 0.0], device=model.device)
                
                cls_loss = cls_loss_fct(diff_logits, target)
                
                # [修改 4] 總 Loss
                # LM Loss (約20) 遠大於 CLS Loss，因此降低 LM 權重，大幅提升 CLS 權重以強調分類準確度
                total_loss = 0.1 * lm_loss + 5.0 * cls_loss
                
                # Backprop
                total_loss = total_loss / args.grad_accum
                total_loss.backward()
                batch_loss += total_loss.item() * args.grad_accum
                
                # Log 顯示 (可以觀察兩個 loss 的變化)
                if step % 20 == 0:
                    writer.add_scalar("train/lm_loss", lm_loss.item(), global_step)
                    writer.add_scalar("train/cls_loss", cls_loss.item(), global_step)
                    writer.add_scalar("train/total_loss", total_loss.item() * args.grad_accum, global_step)

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                global_step += 1
                writer.add_scalar("train/loss", batch_loss, global_step)
                pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

        # Validation
        LOGGER.info(f"Running Validation for Epoch {epoch+1}...")
        val_auc, val_acc, val_f1 = evaluate_auc(model, val_dl, processor, model.device)
        LOGGER.info(f"Epoch {epoch+1} Results | AUC: {val_auc:.4f} | ACC: {val_acc:.4f} | F1: {val_f1:.4f}")
        
        writer.add_scalar("val/auc", val_auc, epoch)
        writer.add_scalar("val/acc", val_acc, epoch)
        writer.add_scalar("val/f1", val_f1, epoch)
        
        if val_auc > best_auc:
            best_auc = val_auc
            # Save only the best model
            # For simplicity, we just overwrite or create a new unique name. 
            # The request says "檔名後面可以記錄分數", implying a folder or file name. 
            # Since `save_pretrained` saves a folder, we will name the folder.
            
            save_name = f"best_model_auc_{best_auc:.4f}_f1_{val_f1:.4f}"
            save_path = os.path.join(log_dir, save_name)
            
            LOGGER.info(f"New Best Model Found! Saving to {save_path}...")
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)

    LOGGER.info("Training Completed.")
    writer.close()

if __name__ == "__main__":
    main()