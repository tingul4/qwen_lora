import os
import glob
import json
import torch
import logging
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# Logging Setup
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("generate_refined")

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

def parse_risk_level(text):
    text = text.lower()
    if "risk level: critical" in text:
        return "Critical"
    if "risk level: high" in text:
        return "High"
    if "risk level: medium" in text:
        return "Medium"
    if "risk level: low" in text:
        return "Low"
    return None

def is_consistent(risk_val, risk_level_str):
    if risk_level_str is None:
        return False
    
    # Risk >= 0.5 -> Critical or High
    if risk_val >= 0.5:
        return risk_level_str in ["Critical", "High"]
    
    # Risk < 0.5 -> Medium or Low
    else:
        return risk_level_str in ["Medium", "Low"]

def generate_candidate(model, processor, frames, user_prompt, temperature, device):
    video_token_str = "<|vision_start|><|video_pad|><|vision_end|>"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": video_token_str}, 
                {"type": "text", "text": user_prompt},
            ],
        }
    ]
    
    # Prepare inputs
    messages_raw = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames}, 
                {"type": "text", "text": user_prompt},
            ],
        }
    ]
    image_inputs, video_inputs = process_vision_info(messages_raw)
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(
        text=[text],
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=512, 
            do_sample=True, 
            temperature=temperature,
            top_p=0.9
        )
        
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    return output_text

def refine_report(model, processor, frames, user_prompt, candidate_report, risk_val, device):
    # Construct conversation history
    video_token_str = "<|vision_start|><|video_pad|><|vision_end|>"
    
    refinery_prompt = f"這張影像 risk={risk_val}，前面生成有問題，請修正成標準格式，不要捏造細節"
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": video_token_str}, 
                {"type": "text", "text": user_prompt},
            ],
        },
        {
            "role": "assistant",
            "content": candidate_report
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": refinery_prompt}
            ]
        }
    ]
    
    # Prepare inputs (need to re-process vision info for the full context? 
    # Actually Qwen2-VL handles multi-turn, but we need to pass the video again or ensure it's linked.
    # In `process_vision_info`, if we pass the video in the first message, it should be fine.
    # But `process_vision_info` expects the raw list.
    
    messages_raw = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames}, 
                {"type": "text", "text": user_prompt},
            ],
        },
        {
            "role": "assistant",
            "content": candidate_report
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": refinery_prompt}
            ]
        }
    ]
    
    image_inputs, video_inputs = process_vision_info(messages_raw)
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(
        text=[text],
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=512, 
            do_sample=False, # Greedy for refinery
        )
        
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    return output_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the Seed LoRA checkpoint")
    parser.add_argument("--csv_path", type=str, default="/raid/mystery-project/dataset/road_train_and_val_with_reports_revised.csv")
    parser.add_argument("--dataset_root", type=str, default="/raid/mystery-project/dataset")
    parser.add_argument("--output_csv", type=str, default="road_train_and_val_pseudo.csv")
    args = parser.parse_args()

    # Load Model
    LOGGER.info("Loading Model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.base_model, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16
    )
    
    LOGGER.info(f"Loading LoRA from {args.lora_path}")
    model = PeftModel.from_pretrained(model, args.lora_path)
    processor = AutoProcessor.from_pretrained(args.base_model)
    
    # Load Data
    df = pd.read_csv(args.csv_path)
    
    # Filter for rows WITHOUT reports
    # But we also need to keep the original rows WITH reports to form the final dataset?
    # The user said: "30 GT + 270 refined pseudo -> 全訓練"
    # So we should process the 270, and then merge.
    
    df_missing = df[df["report"].isna() | (df["report"] == "")].copy()
    LOGGER.info(f"Found {len(df_missing)} samples to generate labels for.")
    
    user_prompt = (
        "分析這張道路影像，輸出格式：\n"
        "Scene Description: ...\n"
        "Risk Level: ... (Critical/High/Medium/Low)\n"
        "Key Hazards: ...\n"
        "Prediction: ..."
    )
    
    results = []
    
    search_dirs = [
        os.path.join(args.dataset_root, "road", "train"),
        os.path.join(args.dataset_root, "road", "test"),
        os.path.join(args.dataset_root, "freeway", "train"),
        os.path.join(args.dataset_root, "freeway", "test")
    ]

    for idx, row in tqdm(df_missing.iterrows(), total=len(df_missing)):
        fname = str(row["file_name"]).strip()
        risk = float(row["risk"])
        
        # Find folder
        folder_path = None
        for d in search_dirs:
            candidate = os.path.join(d, fname)
            if os.path.isdir(candidate):
                folder_path = candidate
                break
        
        if not folder_path:
            LOGGER.warning(f"Folder not found: {fname}")
            continue
            
        # Load Frames
        frame_paths = get_frames_uniformly(folder_path, num_frames=100) # User said "unless memory not enough", but let's try 100 or 32. 
        # Actually user said "影片要使用全採樣，除非memory不夠再改成100" for TRAINING.
        # For generation, let's stick to a reasonable number like 32 or 64 to be fast, or 100 as implied.
        # I'll use 64 to be safe and fast enough.
        frame_paths = get_frames_uniformly(folder_path, num_frames=64)
        
        frames = []
        for p in frame_paths:
            try:
                img = Image.open(p).convert("RGB")
                frames.append(img)
            except Exception:
                pass
        
        if not frames:
            continue

        # Generate Candidates
        candidates = []
        temperatures = [0.7, 0.8, 0.9]
        
        valid_candidates = []
        
        for temp in temperatures:
            cand = generate_candidate(model, processor, frames, user_prompt, temp, model.device)
            candidates.append(cand)
            
            # Filter
            if len(cand) < 100:
                continue
            
            r_level = parse_risk_level(cand)
            if not r_level:
                continue
                
            if is_consistent(risk, r_level):
                valid_candidates.append(cand)
        
        # Select best candidate for refinery
        if valid_candidates:
            # Pick the longest one? Or just the first one?
            # Let's pick the longest one as it might have more detail.
            best_cand = max(valid_candidates, key=len)
        elif candidates:
            # If none valid, pick the longest one from all candidates and try to fix it
            best_cand = max(candidates, key=len)
        else:
            # Should not happen unless generation failed completely
            best_cand = "Generation Failed"
            
        # Refinery
        refined_report = refine_report(model, processor, frames, user_prompt, best_cand, risk, model.device)
        
        results.append({
            "file_name": fname,
            "risk": risk,
            "report": refined_report,
            "original_report": best_cand # Keep for debug
        })
        
        # Save intermediate
        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(args.output_csv, index=False)

    # Final Save
    # Merge with original GT
    df_gt = df[df["report"].notna() & (df["report"] != "")].copy()
    df_pseudo = pd.DataFrame(results)
    
    # Combine
    # Ensure columns match
    df_pseudo = df_pseudo[["file_name", "risk", "report"]]
    df_final = pd.concat([df_gt, df_pseudo], ignore_index=True)
    
    df_final.to_csv(args.output_csv, index=False)
    LOGGER.info(f"Saved final dataset to {args.output_csv} with {len(df_final)} rows.")

if __name__ == "__main__":
    main()
