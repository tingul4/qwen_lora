import os
# Restrict to a single GPU to prevent "device_map='auto'" from spreading across multiple cards
# and causing driver hangs/zombie processes.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from collections import deque
import supervision as sv
from ultralytics import YOLO
from transformers import (
    AutoImageProcessor, 
    AutoModelForDepthEstimation,
    Qwen2VLForConditionalGeneration, 
    AutoProcessor,
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_DIR = os.getenv("VIDEO_DIR", "/ssd6/danielchen/dataset/road/train")
OUTPUT_FILE = "ver3_accident_report.csv"
HISTORY_LEN = 5 # Keep history for smoothing
FPS = 10 # Default FPS since we are reading frames

# Models
YOLO_MODEL = "yolo11x.pt" 
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
QWEN_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

class Pipeline:
    def __init__(self):
        print(f"Initializing Pipeline on {DEVICE}...")
        
        # 1. Detector
        print(f"Loading YOLOv11x ({YOLO_MODEL})...")
        self.detector = YOLO(YOLO_MODEL)
        
        # 2. Depth Estimator
        print(f"Loading Depth Anything V2 ({DEPTH_MODEL_ID})...")
        self.depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).to(DEVICE)
        
        # 3. Tracker
        print("Initializing ByteTrack...")
        self.tracker = sv.ByteTrack()
        
        # 4. Generator (VLM) - Lazy Load
        self.vlm_model = None
        self.vlm_processor = None
        
        # State Tracking
        self.object_history = {} # {track_id: deque([state, ...], maxlen=HISTORY_LEN)}

    def load_vlm(self):
        if self.vlm_model is None:
            print("Loading Qwen2-VL (4-bit)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
                QWEN_MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto"
            )
            self.vlm_processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID, min_pixels=256*28*28, max_pixels=1024*28*28)

    def estimate_depth(self, image_pil):
        inputs = self.depth_processor(images=image_pil, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image_pil.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        return prediction.squeeze().cpu().numpy()

    def get_object_depth(self, box, depth_map):
        x1, y1, x2, y2 = map(int, box)
        # Ensure within bounds
        h, w = depth_map.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        if x1 >= x2 or y1 >= y2:
            return 0.0
            
        # Extract depth crop
        crop = depth_map[y1:y2, x1:x2]
        # Use percentile to filter outliers (e.g., background in the box)
        # 20th percentile is often a good heuristic for the closest point in the box (the object itself)
        # assuming the object is closer than the background.
        return np.percentile(crop, 20)

    def analyze_frames(self, frames_dir):
        # Determine sample rate based on memory
        if torch.cuda.is_available():
            free_mem, _ = torch.cuda.mem_get_info()
            # If > 10GB free, process all frames. Else subsample.
            if free_mem > 10 * 1024**3:
                sample_rate = 1
                print(f"Memory sufficient ({free_mem/1024**3:.1f}GB). Processing ALL frames.")
            else:
                sample_rate = 5
                print(f"Memory tight ({free_mem/1024**3:.1f}GB). Processing every 5th frame.")
        else:
            sample_rate = 5 # CPU fallback

        # List images
        try:
            image_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        except FileNotFoundError:
            print(f"Directory {frames_dir} not found.")
            return None

        if not image_files:
            print(f"No images found in {frames_dir}")
            return None

        self.object_history = {} # Reset history per video
        self.tracker = sv.ByteTrack() # Reset tracker
        
        # Stats to find the "Most Likely Collision Pair" and their "Closest Moment"
        # Structure: { track_id: {'max_score': 0.0, 'min_dist': inf, 'best_frame': None} }
        track_stats = {} 
        
        for frame_idx, img_file in enumerate(image_files):
            if frame_idx % sample_rate != 0:
                continue
                
            img_path = os.path.join(frames_dir, img_file)
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            
            # 1. Detection
            results = self.detector(image_rgb, verbose=False, conf=0.2)[0]
            detections = sv.Detections.from_ultralytics(results)
            
            # 2. Tracking
            detections = self.tracker.update_with_detections(detections)
            
            # 3. Depth Estimation
            depth_map = self.estimate_depth(image_pil)
            
            # 4. Logic & State Update
            current_risks = []
            all_frame_objects = []
            img_h, img_w = depth_map.shape
                        
            for i, (box, _, _, class_id, tracker_id, _) in enumerate(detections):
                class_name = results.names[int(class_id)]
                if tracker_id is None: continue
                track_id = int(tracker_id)
                
                # Get Depth
                d_meter = self.get_object_depth(box, depth_map)
                
                # Spatial Geometry
                x1, y1, x2, y2 = box
                
                # Filter out ego vehicle parts (hood/dashboard)
                # Heuristic: touches bottom of image and is within central region
                if y2 > (img_h * 0.95):
                    bbox_center = (x1 + x2) / 2
                    if (img_w * 0.25) < bbox_center < (img_w * 0.75):
                         continue

                # "Danger Zone" = Central cone of the image (approx ego lane)
                # Use intersection logic instead of center point to catch large vehicles 
                # partially entering the zone (e.g. front in, rear out).
                dz_min = img_w * 0.15
                dz_max = img_w * 0.85
                
                # Check for X-axis overlap
                # Box is [x1, x2], Zone is [dz_min, dz_max]
                # Overlap width = min(x2, dz_max) - max(x1, dz_min)
                intersection_w = max(0, min(x2, dz_max) - max(x1, dz_min))
                
                # Consider in danger zone if intersection is significant (> 10% of box width)
                # This ensures we catch vehicles cutting in, but avoid noise from adjacent lanes just touching the line.
                box_w = x2 - x1
                in_danger_zone = intersection_w > (box_w * 0.1)
                
                # Get State
                state = {
                    "frame": frame_idx,
                    "box": box,
                    "depth": d_meter,
                    "class": class_name,
                    "in_danger_zone": in_danger_zone
                }
                
                # Update History
                if track_id not in self.object_history:
                    self.object_history[track_id] = deque(maxlen=HISTORY_LEN)
                self.object_history[track_id].append(state)
                
                # Calculate Risk Metrics
                history = self.object_history[track_id]
                ttc = float('inf')
                action = "stationary"
                velocity = 0.0
                
                if len(history) >= 2:
                    prev = history[0] # Oldest in window
                    curr = history[-1]
                    
                    dt = (curr["frame"] - prev["frame"]) / FPS
                    d_depth = prev["depth"] - curr["depth"] # Positive if getting closer
                    
                    velocity = d_depth / dt if dt > 0 else 0
                    
                    if velocity > 0.1: # Moving towards ego > 0.1 m/s (lowered from 0.5)
                        action = "approaching"
                        if curr["depth"] > 0:
                            ttc = curr["depth"] / velocity
                    elif velocity < -0.1:
                        action = "receding"
                    
                # Risk Rules
                is_critical = False
                reason = ""
                
                # Rule 1: TTC (Time To Collision)
                ttc_threshold = 3.5 if in_danger_zone else 1.5
                
                # Rule 2: Proximity
                prox_threshold = 25.0 if in_danger_zone else 8.0
                
                if ttc < ttc_threshold:
                    is_critical = True
                    reason = f"Collision Course (TTC={ttc:.1f}s)"
                elif d_meter < prox_threshold and action == "approaching":
                     is_critical = True
                     reason = f"Proximity Warning ({d_meter:.1f}m, Approaching)"
                elif d_meter < (prox_threshold * 0.5) and in_danger_zone:
                     # Stationary but very close in front
                     is_critical = True
                     reason = f"Obstacle Ahead ({d_meter:.1f}m)"
                
                # Format TTC for display
                if ttc == float('inf'):
                    ttc_str = "Static" if abs(velocity) < 0.1 else "Safe"
                else:
                    ttc_str = f"{ttc:.1f}s"

                # --- NEW RISK SCORING ALGORITHM ---
                # Goal: Find the "Oh sh*t" moment, distinguishing crashes from traffic jams.
                
                velocity_factor = max(0, velocity) # We only care if it's approaching
                
                # 1. Impact Potential: (v^2 / d). 
                # High speed at close range is deadly. Low speed at close range is just traffic.
                impact_potential = (velocity_factor ** 2) / max(0.5, d_meter)
                
                # 2. Urgency: 1 / TTC. 
                # Standard metric.
                ttc_valid = ttc if ttc != float('inf') else 100.0
                ttc_factor = 10.0 / max(0.1, ttc_valid)
                
                # 3. Zone Weight.
                # Center lane is much more dangerous.
                zone_factor = 2.0 if in_danger_zone else 0.5
                
                # 4. Congestion Penalty (The Traffic Jam Filter).
                # If relative speed is low (< 1.0 m/s or 3.6 km/h), it's likely just following traffic or a stop light.
                # Unless it's EXTREMELY close (< 2m), then it might be a "bump".
                if velocity_factor < 1.0:
                    if d_meter < 2.0: 
                        congestion_penalty = 0.5 # Still somewhat risky if literally touching
                    else:
                        congestion_penalty = 0.1 # Ignore normal traffic proximity
                else:
                    congestion_penalty = 1.0
                
                obj_score = (impact_potential + ttc_factor) * zone_factor * congestion_penalty

                obj_data = {
                    "id": track_id,
                    "class": class_name,
                    "distance": f"{d_meter:.1f}m",
                    "raw_dist": d_meter,
                    "action": action,
                    "ttc": ttc_str,
                    "reason": reason,
                    "box": box,
                    "in_danger_zone": in_danger_zone,
                    "risk_score": obj_score
                }
                
                all_frame_objects.append(obj_data)
                
                if is_critical:
                    current_risks.append(obj_data)
            
            # If risks found, process this frame as a candidate
            if current_risks:
                ego_state = "moving" 
                
                # Frame Score (for reference, though we use track logic now)
                frame_risk_score = max([obj['risk_score'] for obj in current_risks]) if current_risks else 0.0

                scene_desc = {
                    "frame_id": frame_idx,
                    "critical_objects": current_risks,
                    "all_objects": all_frame_objects,
                    "ego_state": ego_state,
                    "frame_risk_score": frame_risk_score
                }
                
                frame_package = {
                    "frame_idx": frame_idx,
                    "image": image_pil, 
                    "json_desc": scene_desc
                }
                
                # --- UPDATE GLOBAL TRACK STATISTICS ---
                # We want to find the frame where the "Highest Risk Object" is "Closest".
                for obj in current_risks:
                    tid = obj['id']
                    score = obj['risk_score']
                    dist = obj['raw_dist']
                    
                    if tid not in track_stats:
                        track_stats[tid] = {'max_score': -1.0, 'min_dist': float('inf'), 'best_frame': None}
                    
                    # 1. Track the MAX risk this object ever reached (to identify IF it is the main threat)
                    if score > track_stats[tid]['max_score']:
                        track_stats[tid]['max_score'] = score
                    
                    # 2. Track the MIN distance and save THAT frame (to capture the moment of impact/closest approach)
                    # Note: We only save the frame if it's the closest approach SO FAR.
                    if dist < track_stats[tid]['min_dist']:
                        track_stats[tid]['min_dist'] = dist
                        track_stats[tid]['best_frame'] = frame_package
            
        # --- SELECTION LOGIC ---
        if not track_stats:
            return None
            
        # 1. Identify the "Most Dangerous Object"
        # The object that achieved the highest risk score at any point in the video.
        most_dangerous_tid = max(track_stats, key=lambda x: track_stats[x]['max_score'])
        
        print(f"  Selected Primary Threat ID: {most_dangerous_tid} (Max Score: {track_stats[most_dangerous_tid]['max_score']:.1f})")
        print(f"  Selecting Frame at Min Dist: {track_stats[most_dangerous_tid]['min_dist']:.1f}m")
        
        # 2. Return the frame where THAT object was closest
        best_frame = track_stats[most_dangerous_tid]['best_frame']
        return best_frame

    def generate_report_prompt(self, scene_json):
        # Convert JSON to Natural Language
        desc = f"Current Scene Telemetry (Frame {scene_json['frame_id']}):\n"
        for obj in scene_json['critical_objects']:
            desc += f"- A {obj['class']} is {obj['distance']} away, {obj['action']} with a Time-to-Collision of {obj['ttc']}. Warn: {obj['reason']}.\n"
        desc += f"Ego Vehicle State: {scene_json['ego_state']}.\n"
        desc += "Task: Generate a risk assessment report warning the driver."
        return desc

    def run_vlm_inference(self, frame_data):
        self.load_vlm()
        
        prompt_text = self.generate_report_prompt(frame_data['json_desc'])
        
        # Visualize for debug/saving
        # (Optional: Draw boxes on image before passing to VLM? Usually better to give clean image + text)
        # But we can save a visualized version locally
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame_data["image"]},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.vlm_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)
        
        with torch.no_grad():
            generated_ids = self.vlm_model.generate(
                **inputs, 
                max_new_tokens=512,
                do_sample=True,
                temperature=0.4,
                top_p=0.9
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.vlm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text, prompt_text

def main():
    pipeline = Pipeline()
    
    # Read dataset
    if os.path.exists("road_train_and_val_annot.csv"):
        df = pd.read_csv("road_train_and_val_annot.csv")
    elif os.path.exists("ver2/road_train_and_val_annot.csv"):
        df = pd.read_csv("ver2/road_train_and_val_annot.csv")
    elif os.path.exists("../ver2/road_train_and_val_annot.csv"):
        df = pd.read_csv("../ver2/road_train_and_val_annot.csv")
    else:
        print("Annotation file not found.")
        return

    results = []
    
    # Process sample
    for idx, row in tqdm(df.head(5).iterrows(), total=5):
        file_name = row['file_name']
        frames_dir = os.path.join(VIDEO_DIR, file_name)
        
        if not os.path.exists(frames_dir):
            print(f"Frames directory {frames_dir} not found.")
            continue
            
        print(f"Processing {file_name}...")
        critical_frame = pipeline.analyze_frames(frames_dir)
        
        if critical_frame:
            print(f"Detected Critical Frame: {critical_frame['frame_idx']}")
            report, prompt = pipeline.run_vlm_inference(critical_frame)
            
            # Save Viz
            vis_dir = "ver3_visualizations"
            os.makedirs(vis_dir, exist_ok=True)
            vis_path = os.path.join(vis_dir, f"{file_name}_critical.jpg")
            
            # Draw boxes
            img_cv = cv2.cvtColor(np.array(critical_frame['image']), cv2.COLOR_RGB2BGR)
            h, w, _ = img_cv.shape
            
            # Draw Danger Zone (Central 70% -> 15% to 85%? No, previous logic was center +/- 35% which is 15% to 85%)
            # Logic: dist_from_center < (img_w * 0.35). Center is 0.5. 
            # Left bound: 0.5 - 0.35 = 0.15. Right bound: 0.5 + 0.35 = 0.85.
            dz_x1 = int(w * 0.15)
            dz_x2 = int(w * 0.85)
            cv2.line(img_cv, (dz_x1, 0), (dz_x1, h), (0, 255, 255), 1) # Yellow Line
            cv2.line(img_cv, (dz_x2, 0), (dz_x2, h), (0, 255, 255), 1)
            
            # Draw ALL objects (Gray)
            if 'all_objects' in critical_frame['json_desc']:
                for obj in critical_frame['json_desc']['all_objects']:
                    x1, y1, x2, y2 = map(int, obj['box'])
                    # Gray color
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (128, 128, 128), 1)
                    cv2.putText(img_cv, f"{obj['class']} {obj['distance']}", (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

            # Draw CRITICAL objects (Red) - Overwrite gray
            for obj in critical_frame['json_desc']['critical_objects']:
                x1, y1, x2, y2 = map(int, obj['box'])
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                label = f"WARN: {obj['class']} TTC:{obj['ttc']}"
                cv2.putText(img_cv, label, (x1, y1-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                reason = obj.get('reason', '')
                cv2.putText(img_cv, reason, (x1, y2+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                           
            cv2.imwrite(vis_path, img_cv)
            
            results.append({
                "file_name": file_name,
                "prompt": prompt,
                "report": report,
                "frame_id": critical_frame['frame_idx']
            })
            print(f"Report:\n{report}\n")
        else:
            print(f"No critical risks detected in {file_name}")

    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f"Saved results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()