import pandas as pd
import json
import os
import cv2
from tqdm import tqdm

def extract_keyframe(video_path, output_path):
    """Extracts the middle frame from the video."""
    if os.path.exists(output_path):
        return True
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
        
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame = frame_count // 2
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ret, frame = cap.read()
    
    if ret:
        cv2.imwrite(output_path, frame)
    
    cap.release()
    return ret

def build_qwen_dataset():
    # Paths
    csv_path = "/raid/mystery-project/dataset/road_train_and_val_with_reports_revised.csv"
    sam_json_path = "/raid/mystery-project/qwen_lora/sam3_qwen_pipeline/sam3_summaries.json"
    video_dir = "/raid/mystery-project/dataset/road/videos"
    image_dir = "/raid/mystery-project/dataset/road/keyframes"
    output_jsonl = "/raid/mystery-project/qwen_lora/sam3_qwen_pipeline/qwen_finetune_data.jsonl"
    
    os.makedirs(image_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    
    # Load SAM3 summaries (Mock if not exists for now)
    if os.path.exists(sam_json_path):
        with open(sam_json_path, 'r') as f:
            sam_summaries = json.load(f)
    else:
        print("Warning: SAM3 summaries not found. Using placeholders.")
        sam_summaries = {}

    dataset_items = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        video_name = row['file_name']
        risk_gt = row['risk']
        report_gt = row['report']
        
        # 1. Prepare Image
        video_path = os.path.join(video_dir, f"{video_name}.mp4")
        if not os.path.exists(video_path):
             video_path = os.path.join(video_dir, f"{video_name}") # Try without extension
             
        image_path = os.path.join(image_dir, f"{video_name}.jpg")
        
        if os.path.exists(video_path):
            if not extract_keyframe(video_path, image_path):
                continue # Skip if extraction fails
        else:
            # If video missing, skip or use placeholder if testing
            continue

        # 2. Prepare Text Input
        sam_summary = sam_summaries.get(video_name, "No specific object relations detected.")
        
        system_prompt = "You are an expert traffic safety analyst. Analyze the image and the provided object tracking data."
        
        # Check if we have a Report GT (Source A)
        has_report = isinstance(report_gt, str) and len(report_gt) > 10
        
        conversations = []
        
        if has_report:
            # Task Type: Full Report Generation
            user_text = f"Object Tracking Data:\n{sam_summary}\n\nAnalysis:"
            assistant_text = report_gt
        else:
            # Task Type: Risk Classification (Source B)
            user_text = f"Object Tracking Data:\n{sam_summary}\n\nAssess the risk level."
            risk_level = "High" if risk_gt == 1 else "Low" # Simplified mapping
            assistant_text = f"Risk Level: {risk_level}"

        # Construct Qwen-VL format (Standard Chat Format)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"{image_path}"},
                    {"type": "text", "text": user_text}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_text}
                ]
            }
        ]
        
        item = {
            "id": f"identity_{idx}",
            "conversations": messages
        }
        dataset_items.append(item)

    # Save to JSONL
    with open(output_jsonl, 'w') as f:
        for item in dataset_items:
            f.write(json.dumps(item) + '\n')
            
    print(f"Dataset built with {len(dataset_items)} samples. Saved to {output_jsonl}")

if __name__ == "__main__":
    build_qwen_dataset()
