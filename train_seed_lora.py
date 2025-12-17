import os
import glob
import random
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
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
LOGGER = logging.getLogger("train_seed")

# ===================== Utils =====================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_frames_uniformly(folder_path, num_frames=8):
    exts = ["*.jpg", "*.png", "*.jpeg", "*.JPG"]
    all_files = []
    for e in exts:
        all_files.extend(glob.glob(os.path.join(folder_path, e)))
    all_files = sorted(all_files)
    
    total_frames = len(all_files)
    if total_frames == 0:
        return []
    
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

def load_seed_data(dataset_root, csv_path):
    """
    Load only the 30 samples with reports from the CSV.
    Split into 25 train and 5 val.
    """
    if not os.path.exists(csv_path):
        raise ValueError(f"CSV not found: {csv_path}")
        
    df = pd.read_csv(csv_path)
    LOGGER.info(f"Loading data from: {csv_path} ({len(df)} rows)")
    
    labeled_data = []
    
    # Search paths
    search_dirs = [
        os.path.join(dataset_root, "road", "train"),
        os.path.join(dataset_root, "road", "test"), # Just in case
        os.path.join(dataset_root, "freeway", "train"),
        os.path.join(dataset_root, "freeway", "test")
    ]

    for _, row in df.iterrows():
        report = row.get("report", "")
        if pd.isna(report) or str(report).strip() == "":
            continue
            
        fname = str(row["file_name"]).strip()
        risk = float(row["risk"])
        
        # Find the folder
        full_path = None
        for d in search_dirs:
            candidate = os.path.join(d, fname)
            if os.path.isdir(candidate):
                full_path = candidate
                break
        
        if full_path:
            labeled_data.append({
                "folder_path": full_path,
                "file_name": fname,
                "risk": risk,
                "report": str(report).strip()
            })
        else:
            LOGGER.warning(f"Folder not found for {fname}")

    LOGGER.info(f"Total Labeled Samples Found: {len(labeled_data)}")
    
    # Shuffle and split 25/5
    random.shuffle(labeled_data)
    train_data = labeled_data[:25]
    val_data = labeled_data[25:]
    
    LOGGER.info(f"Split: {len(train_data)} Train, {len(val_data)} Val")
    return train_data, val_data

# ===================== Dataset =====================
class SeedDataset(Dataset):
    def __init__(self, data_list, processor, num_frames=-1):
        self.data = data_list
        self.processor = processor
        self.num_frames = num_frames
        
        # Updated Prompt
        self.user_prompt = (
            "分析這張道路影像，輸出格式：\n"
            "Scene Description: ...\n"
            "Risk Level: ... (Critical/High/Medium/Low)\n"
            "Key Hazards: ...\n"
            "Prediction: ..."
        )
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        folder_path = item["folder_path"]
        report = item["report"]
        
        # 1. Load Images
        frame_paths = get_frames_uniformly(folder_path, num_frames=self.num_frames)
        if len(frame_paths) == 0:
            # Fallback to random sample if empty (shouldn't happen if data is clean)
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

        # 2. Prepare Messages
        video_token_str = "<|vision_start|><|video_pad|><|vision_end|>"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": video_token_str}, 
                    {"type": "text", "text": self.user_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": report
            }
        ]

        # 3. Process Vision
        # We need a dummy messages list for process_vision_info to extract pixel values
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
                "content": report
            }
        ]
        image_inputs, video_inputs = process_vision_info(messages_raw)

        # 4. Apply Chat Template
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        # 5. Tokenize
        inputs = self.processor(
            text=[text],
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Labels for LM loss
        inputs["labels"] = inputs["input_ids"].clone()
        if "attention_mask" in inputs:
            inputs["labels"][inputs["attention_mask"] == 0] = -100

        # Squeeze batch dim
        for k, v in inputs.items():
            if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == 1:
                inputs[k] = v.squeeze(0)
                
        if "pixel_values_videos" not in inputs:
             return self.__getitem__(random.randint(0, len(self.data) - 1))

        return inputs

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return batch 

# ===================== Main =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="/raid/mystery-project/dataset")
    parser.add_argument("--csv_path", type=str, default="/raid/mystery-project/dataset/road_train_and_val_with_reports_revised.csv")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4) # Slightly higher LR for small dataset/LoRA
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=-1, help="-1 for all frames")
    parser.add_argument("--output_dir", type=str, default="checkpoints_seed_lora")
    args = parser.parse_args()

    set_seed(42)
    
    # Logging
    timestamp = datetime.now().strftime("%m%d%H%M")
    log_dir = os.path.join("logs", "seed_" + timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "train.log")),
            logging.StreamHandler()
        ]
    )
    global LOGGER
    LOGGER = logging.getLogger("train_seed")
    LOGGER.info(f"Logging to: {log_dir}")
    
    writer = SummaryWriter(log_dir=log_dir)

    # 1. Load Data
    train_data, val_data = load_seed_data(args.dataset_root, args.csv_path)
    
    # 2. Model & Processor
    LOGGER.info("Loading Model...")
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

    # 3. LoRA
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Dataloaders
    train_ds = SeedDataset(train_data, processor, num_frames=args.num_frames)
    val_ds = SeedDataset(val_data, processor, num_frames=args.num_frames)
    
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # 5. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_dl) * args.epochs // args.grad_accum
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)

    # 6. Training Loop
    global_step = 0
    min_val_loss = float("inf")
    
    LOGGER.info("Starting Seed LoRA Training...")
    optimizer.zero_grad()
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step, batch in enumerate(pbar):
            # batch is a list of samples because of collate_fn
            batch_loss = 0.0
            
            for sample in batch:
                # Move to device
                input_ids = sample["input_ids"].unsqueeze(0).to(model.device)
                attention_mask = sample["attention_mask"].unsqueeze(0).to(model.device)
                pixel_values_videos = sample["pixel_values_videos"].to(model.device)
                video_grid_thw = sample["video_grid_thw"].unsqueeze(0).to(model.device)
                labels = sample["labels"].unsqueeze(0).to(model.device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values_videos=pixel_values_videos,
                    video_grid_thw=video_grid_thw,
                    labels=labels
                )
                
                loss = outputs.loss
                loss = loss / args.grad_accum
                loss.backward()
                batch_loss += loss.item() * args.grad_accum
            
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                writer.add_scalar("train/loss", batch_loss, global_step)
                pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                for sample in batch:
                    input_ids = sample["input_ids"].unsqueeze(0).to(model.device)
                    attention_mask = sample["attention_mask"].unsqueeze(0).to(model.device)
                    pixel_values_videos = sample["pixel_values_videos"].to(model.device)
                    video_grid_thw = sample["video_grid_thw"].unsqueeze(0).to(model.device)
                    labels = sample["labels"].unsqueeze(0).to(model.device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values_videos=pixel_values_videos,
                        video_grid_thw=video_grid_thw,
                        labels=labels
                    )
                    val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_dl)
        LOGGER.info(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar("val/loss", avg_val_loss, epoch)
        
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            save_path = os.path.join(log_dir, "best_seed_model")
            LOGGER.info(f"New Best Model! Saving to {save_path}")
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)

    LOGGER.info("Seed Training Completed.")
    writer.close()

if __name__ == "__main__":
    main()
