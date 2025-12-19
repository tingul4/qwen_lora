import os
import json
import pandas as pd
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

def run_inference():
    # Paths
    base_model_id = "Qwen/Qwen2-VL-7B-Instruct"
    # Point to the last checkpoint saved by finetune.py
    adapter_path = "/raid/mystery-project/qwen_lora/sam3_qwen_pipeline/checkpoints/epoch_10" 
    csv_path = "/raid/mystery-project/dataset/road_train_and_val_with_reports_revised.csv"
    sam_json_path = "/raid/mystery-project/qwen_lora/sam3_qwen_pipeline/sam3_summaries.json"
    image_dir = "/raid/mystery-project/dataset/road/keyframes"
    output_csv = "/raid/mystery-project/qwen_lora/sam3_qwen_pipeline/generated_reports_full.csv"

    # Load SAM summaries
    if os.path.exists(sam_json_path):
        with open(sam_json_path, 'r') as f:
            sam_summaries = json.load(f)
    else:
        print("Warning: SAM summaries not found.")
        sam_summaries = {}

    # Load Processor
    processor = AutoProcessor.from_pretrained(base_model_id, min_pixels=256*28*28, max_pixels=1024*28*28)

    # Load Base Model with QLoRA (4-bit) to save memory
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    print("Loading Base Model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_id, 
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Load LoRA adapter
    if os.path.exists(adapter_path):
        print(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        print(f"Warning: LoRA adapter not found at {adapter_path}. Running with base model (Zero-shot).")

    df = pd.read_csv(csv_path)
    results = []

    print("Starting Inference...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        video_name = row['file_name']
        image_path = os.path.join(image_dir, f"{video_name}.jpg")
        
        # Skip if image doesn't exist (maybe video was missing)
        if not os.path.exists(image_path):
            continue

        sam_summary = sam_summaries.get(video_name, "No specific object relations detected.")
        
        # Construct Prompt (Same format as training)
        user_text = f"Object Tracking Data:\n{sam_summary}\n\nAnalysis:"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": user_text}
                ]
            }
        ]

        # Prepare inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        results.append({
            "file_name": video_name,
            "generated_report": output_text,
            "gt_risk": row['risk'],
            "gt_report": row['report'] if 'report' in row else ""
        })

    # Save results
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)
    print(f"Saved generated reports to {output_csv}")

if __name__ == "__main__":
    run_inference()
