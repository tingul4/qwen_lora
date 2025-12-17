import os
import cv2
import numpy as np
import pandas as pd
import json
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor

# Check for GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {DEVICE}")

class SAM3Wrapper:
    """
    Implements 'SAM3' (Perception Layer) using Hugging Face Transformers.
    Since 'SAM3' (Text-to-Mask in Video) is not a single standard model yet,
    we implement it as a pipeline:
    1. Grounding DINO: Text Prompts -> Bounding Boxes
    2. SAM (Segment Anything): Bounding Boxes -> Masks
    """
    def __init__(self):
        print("Initializing Perception Models (Grounding DINO + SAM)...")
        
        # 1. Grounding DINO (Text -> Box)
        self.dino_id = "IDEA-Research/grounding-dino-base"
        self.dino_processor = AutoProcessor.from_pretrained(self.dino_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.dino_id).to(DEVICE)
        
        # 2. SAM (Box -> Mask)
        # Using SAM-ViT-Base for speed, can switch to 'facebook/sam-vit-huge' for quality
        self.sam_id = "facebook/sam-vit-base" 
        self.sam_processor = SamProcessor.from_pretrained(self.sam_id)
        self.sam_model = SamModel.from_pretrained(self.sam_id).to(DEVICE)
        
        print("Models loaded successfully.")

    def predict(self, frame, prompts):
        """
        Args:
            frame: numpy array (H, W, C) - BGR format from cv2
            prompts: list of strings e.g. ["scooter", "car"]
        Returns:
            List of dicts with 'id', 'label', 'mask', 'bbox', 'center'
        """
        # Convert BGR (OpenCV) to RGB (PIL)
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # --- Step 1: Detection (Grounding DINO) ---
        # Format prompts for Grounding DINO (dot separated)
        text_prompt = ". ".join(prompts) + "."
        
        inputs = self.dino_processor(images=image_pil, text=text_prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
            
        # Post-process boxes
        # target_sizes must be a tensor
        target_sizes = torch.tensor([image_pil.size[::-1]], device=DEVICE)
        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            threshold=0.35,
            text_threshold=0.35,
            target_sizes=target_sizes
        )[0]
        
        boxes = results["boxes"]
        scores = results["scores"]
        labels = results["labels"] # These are indices, need to map back if needed
        
        # Filter and map labels
        # Grounding DINO returns label indices relative to the prompt tokens, which is complex to map back directly 
        # without the tokenizer. For simplicity in this pipeline, we will treat all detections as "object" 
        # or try to infer based on prompt order if needed. 
        # A robust way is to check the phrase corresponding to the token.
        # For this "Mystery Project", we will simplify:
        # We assume the model detects what we asked. We can assign labels based on simple logic or just "detected_object".
        
        detected_objects = []
        
        if len(boxes) > 0:
            # --- Step 2: Segmentation (SAM) ---
            # SAM takes boxes as prompts
            # Prepare inputs for SAM
            # SAM expects inputs in a specific format. The processor handles resizing.
            sam_inputs = self.sam_processor(
                image_pil, 
                input_boxes=[[boxes.tolist()]], # Batch size 1, list of boxes
                return_tensors="pt"
            ).to(DEVICE)
            
            with torch.no_grad():
                sam_outputs = self.sam_model(**sam_inputs)
            
            # Post-process masks
            # SAM returns masks: (batch, num_boxes, 3, H, W) - 3 masks per box (multimask)
            # We usually take the one with highest score, or just the first one.
            sam_masks = self.sam_processor.image_processor.post_process_masks(
                sam_outputs.pred_masks, 
                sam_inputs["original_sizes"], 
                sam_inputs["reshaped_input_sizes"]
            )[0] # Shape: (num_boxes, 3, H, W)
            
            # Iterate over detections
            for i in range(len(boxes)):
                box = boxes[i].cpu().numpy() # [x_min, y_min, x_max, y_max]
                score = scores[i].item()
                
                # Get best mask (index 0 usually best for single object)
                mask = sam_masks[i, 0].cpu().numpy() # (H, W) bool
                
                # Calculate center
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                
                # Determine label
                # With post_process_grounded_object_detection, labels are usually the text phrases
                if len(labels) > i and isinstance(labels[i], str):
                    label = labels[i]
                else:
                    label = "vehicle"
 
                
                detected_objects.append({
                    'id': i,
                    'label': label,
                    'score': score,
                    'bbox': box,
                    'center': (center_x, center_y),
                    'mask': mask.astype(np.uint8)
                })
                
        return detected_objects

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def analyze_video_relations(video_path, sam_model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video."

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_data = []
    frame_idx = 0
    
    # Process every 5th frame to save compute
    skip_frames = 5
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % skip_frames == 0:
            # 1. Object Tracking/Detection
            objects = sam_model.predict(frame, ["scooter", "car", "truck", "pedestrian"])
            
            # Normalize coordinates
            for obj in objects:
                obj['center_norm'] = (obj['center'][0] / width, obj['center'][1] / height)
            
            frame_data.append({
                'frame': frame_idx,
                'objects': objects
            })
            
        frame_idx += 1
    
    cap.release()
    
    # 2. Calculate Relations (Heuristics)
    summary_events = []
    
    for i in range(1, len(frame_data)):
        prev_frame = frame_data[i-1]
        curr_frame = frame_data[i]
        
        curr_objects = curr_frame['objects']
        prev_objects = prev_frame['objects']
        
        # Map objects by ID (assuming SAM3 provides consistent tracking IDs)
        prev_obj_map = {obj['id']: obj for obj in prev_objects}
        
        risk_score = 0
        
        for obj in curr_objects:
            # Check for Lane Crossing (Lateral movement)
            if obj['id'] in prev_obj_map:
                prev_obj = prev_obj_map[obj['id']]
                lateral_move = abs(obj['center_norm'][0] - prev_obj['center_norm'][0])
                if lateral_move > 0.05: # Threshold for rapid lane change
                    summary_events.append(f"Frame {curr_frame['frame']}: {obj['label']} (ID {obj['id']}) is changing lanes rapidly.")
            
            # Check for Proximity with other objects
            for other_obj in curr_objects:
                if obj['id'] < other_obj['id']: # Avoid duplicates
                    dist = calculate_distance(obj['center_norm'], other_obj['center_norm'])
                    
                    # Heuristic: Approach Rate
                    approach_rate = 0
                    if obj['id'] in prev_obj_map and other_obj['id'] in prev_obj_map:
                        prev_dist = calculate_distance(prev_obj_map[obj['id']]['center_norm'], 
                                                     prev_obj_map[other_obj['id']]['center_norm'])
                        approach_rate = prev_dist - dist
                    
                    if dist < 0.1: # Very close
                        risk_msg = f"Frame {curr_frame['frame']}: {obj['label']} and {other_obj['label']} are critically close (dist={dist:.2f})."
                        if approach_rate > 0.01:
                            risk_msg += " Closing in fast!"
                        summary_events.append(risk_msg)

    # 3. Generate Structured Summary
    if not summary_events:
        return "No significant risk events detected."
    
    # Deduplicate and summarize
    unique_events = sorted(list(set(summary_events)), key=lambda x: int(x.split(' ')[1].replace(':', '')))
    
    # Limit to top events to avoid context overflow
    structured_summary = "SAM3 Perception Summary:\n" + "\n".join(unique_events[:20])
    
    return structured_summary

def main():
    # Paths
    video_dir = "/raid/mystery-project/dataset/road/videos"
    csv_path = "/raid/mystery-project/dataset/road_train_and_val_with_reports_revised.csv"
    output_json = "/raid/mystery-project/qwen_lora/sam3_qwen_pipeline/sam3_summaries.json"
    
    df = pd.read_csv(csv_path)
    sam_model = SAM3Wrapper()
    
    results = {}
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_name = row['file_name']
        # Handle extension if needed
        video_path = os.path.join(video_dir, f"{video_name}.mp4") 
        if not os.path.exists(video_path):
             # Try without extension or other formats
             video_path = os.path.join(video_dir, f"{video_name}")
        
        if os.path.exists(video_path):
            summary = analyze_video_relations(video_path, sam_model)
            results[video_name] = summary
        else:
            results[video_name] = "Video not found."
            
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved summaries to {output_json}")

if __name__ == "__main__":
    main()
