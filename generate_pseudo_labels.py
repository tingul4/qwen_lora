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
LOGGER = logging.getLogger("generate_labels")

def get_frames_uniformly(folder_path, num_frames=8):
    exts = ["*.jpg", "*.png", "*.jpeg", "*.JPG"]
    all_files = []
    for e in exts:
        all_files.extend(glob.glob(os.path.join(folder_path, e)))
    all_files = sorted(all_files)
    
    total_frames = len(all_files)
    if total_frames == 0:
        return []
    
    if total_frames < num_frames:
        return [all_files[i % total_frames] for i in range(num_frames)]
    
    indices = np.linspace(0, total_frames - 1, num_frames + 1, dtype=int)
    selected_paths = []
    for i in range(num_frames):
        idx = (indices[i] + indices[i+1]) // 2 
        selected_paths.append(all_files[idx])
        
    return selected_paths

def generate_report(model, processor, folder_path, user_prompt, device):
    frames = []
    frame_paths = get_frames_uniformly(folder_path, num_frames=8)
    if not frame_paths:
        return None

    for p in frame_paths:
        try:
            img = Image.open(p).convert("RGB")
            frames.append(img)
        except Exception:
            pass
    
    if not frames:
        return None

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": user_prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    inputs = inputs.to(device)

    # Generate
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text

def process_csv(csv_path, dataset_root, subfolder, model, processor, device, output_path):
    if not os.path.exists(csv_path):
        LOGGER.warning(f"CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    LOGGER.info(f"Processing {csv_path} ({len(df)} rows)")
    
    updated_count = 0
    skipped_count = 0
    
    # Check if 'report' column exists, if not create it
    if "report" not in df.columns:
        df["report"] = ""
    
    # Ensure report column is string type to avoid issues
    df["report"] = df["report"].astype(str)

    # user_prompt = "Is there an accident or potential danger in this video? Answer only with 'Yes' or 'No'."
    user_prompt = "generate the analysis of the road condition and conclude whether there is an accident or potential danger in this video. use below type: {Scene Description: \nRisk Level: \nKey Hazards: \nPrediction: }"

    # Note: The model is trained to output "Analysis: ...\nConclusion: ..."
    # We want to extract the Analysis part.

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        current_report = str(row["report"]).strip()
        risk = float(row["risk"])
        
        # If report is missing or empty or 'nan', generate it
        if not current_report or current_report.lower() == "nan":
            fname = str(row["file_name"]).strip()
            folder_path = os.path.join(dataset_root, subfolder, "train", fname)
            
            if not os.path.exists(folder_path):
                # Try test folder if not in train (though this script is for training data usually)
                folder_path = os.path.join(dataset_root, subfolder, "test", fname)
            
            if os.path.exists(folder_path):
                generated_text = generate_report(model, processor, folder_path, user_prompt, device)
                
                if generated_text:
                    # Debug: Print first few outputs to see what's happening
                    if updated_count + skipped_count < 5:
                        LOGGER.info(f"--- Debug Sample {fname} ---")
                        LOGGER.info(f"GT Risk: {risk}")
                        LOGGER.info(f"Generated: {generated_text}")
                        LOGGER.info("-----------------------------")

                    # Parse the output
                    # Expected format: "Analysis: ...\nConclusion: ..."
                    
                    analysis_text = ""
                    conclusion = ""
                    
                    # Robust Parsing Strategy
                    # 1. Try to find Conclusion first
                    if "Conclusion:" in generated_text:
                        parts = generated_text.split("Conclusion:", 1)
                        analysis_part = parts[0].strip()
                        conclusion = parts[1].strip().lower()
                        
                        # Remove "Analysis:" prefix from analysis_part if exists
                        if "Analysis:" in analysis_part:
                            analysis_part = analysis_part.split("Analysis:", 1)[1].strip()
                        
                        analysis_text = analysis_part
                    else:
                        # If no explicit Conclusion tag, try to infer from the end or just fail consistency
                        # But we can still try to see if the model just outputted the analysis
                        analysis_text = generated_text.replace("Analysis:", "").strip()
                        # Try to find Yes/No in the text if it's short? 
                        # Or maybe the model failed to follow format.
                        # Let's check if "Yes" or "No" is at the end.
                        lower_text = generated_text.lower()
                        if lower_text.endswith("yes") or lower_text.endswith("yes."):
                            conclusion = "yes"
                        elif lower_text.endswith("no") or lower_text.endswith("no."):
                            conclusion = "no"
                    
                    # ==========================================
                    # Label Refinery: Save All Generated Reports
                    # ==========================================
                    # User requested to skip consistency check for now.
                    # We save the analysis text regardless of the conclusion.
                    
                    if analysis_text:
                        df.at[idx, "report"] = analysis_text
                        updated_count += 1
                    else:
                        skipped_count += 1
                        
            else:
                LOGGER.warning(f"Folder not found: {folder_path}")

    LOGGER.info(f"Updated {updated_count} reports. Skipped {skipped_count} inconsistent reports.")
    df.to_csv(output_path, index=False)
    LOGGER.info(f"Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="/raid/mystery-project/dataset")
    parser.add_argument("--freeway_train_csv", type=str, default=None)
    parser.add_argument("--road_train_csv", type=str, default="/raid/mystery-project/dataset/road_train_with_reports.csv")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the trained LoRA adapter")
    parser.add_argument("--output_dir", type=str, default="refined_labels")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    # Load Model
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
    
    # Load LoRA Adapter
    LOGGER.info(f"Loading Adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    
    processor = AutoProcessor.from_pretrained(args.model_id)
    
    # Process Freeway
    if args.freeway_train_csv is not None:
        freeway_out = os.path.join(args.output_dir, "freeway_train_refined.csv")
        process_csv(args.freeway_train_csv, args.dataset_root, "freeway", model, processor, model.device, freeway_out)
    
    # Process Road
    if args.road_train_csv is not None:
        road_out = os.path.join(args.output_dir, "road_train_refined.csv")
        process_csv(args.road_train_csv, args.dataset_root, "road", model, processor, model.device, road_out)

if __name__ == "__main__":
    main()
