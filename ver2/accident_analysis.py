import os
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor, 
    AutoModelForZeroShotObjectDetection, 
    SamModel, 
    SamProcessor,
    AutoImageProcessor, 
    AutoModelForDepthEstimation,
    Qwen2VLForConditionalGeneration,
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_DIR = "/raid/mystery-project/dataset/road/videos"
OUTPUT_FILE = "accident_report.csv"

class SAM3Wrapper:
    def __init__(self):
        print("Initializing SAM3 (Grounding DINO + SAM)...")
        self.dino_id = "IDEA-Research/grounding-dino-base"
        self.dino_processor = AutoProcessor.from_pretrained(self.dino_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.dino_id).to(DEVICE)
        
        self.sam_id = "facebook/sam-vit-base"
        self.sam_processor = SamProcessor.from_pretrained(self.sam_id)
        self.sam_model = SamModel.from_pretrained(self.sam_id).to(DEVICE)

    def predict(self, image_pil, prompts=["car", "truck", "bus", "motorcycle", "pedestrian"]):
        # Grounding DINO
        text = ". ".join(prompts) + "."
        inputs = self.dino_processor(images=image_pil, text=text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
        
        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.3,
            text_threshold=0.25,
            target_sizes=[image_pil.size[::-1]]
        )[0]

        if len(results["boxes"]) == 0:
            return []

        # SAM
        inputs = self.sam_processor(image_pil, input_boxes=[results["boxes"].tolist()], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.sam_model(**inputs)
        
        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )[0]
        
        objects = []
        for i in range(len(results["boxes"])):
            objects.append({
                "label": results["labels"][i],
                "score": results["scores"][i].item(),
                "box": results["boxes"][i].tolist(), # [xmin, ymin, xmax, ymax]
                "mask": masks[i][0].numpy() # First mask
            })
        return objects

class DepthEstimator:
    def __init__(self):
        print("Initializing Depth Anything...")
        self.model_id = "LiheYoung/depth-anything-small-hf"
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id).to(DEVICE)

    def predict(self, image_pil):
        inputs = self.processor(images=image_pil, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image_pil.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        return prediction.squeeze().cpu().numpy()

class AccidentAnalyzer:
    def __init__(self):
        self.sam = SAM3Wrapper()
        self.depth_model = DepthEstimator()
        self.vlm_model = None
        self.vlm_processor = None

    def load_vlm(self):
        if self.vlm_model is None:
            print("Loading Qwen2-VL (4-bit)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct",
                quantization_config=bnb_config,
                device_map="auto"
            )
            self.vlm_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=256*28*28, max_pixels=1024*28*28)

    def get_object_depth(self, obj, depth_map):
        # Calculate average depth within the mask
        mask = obj["mask"]
        # Use median to be more robust to outliers at edges
        obj_depth = np.median(depth_map[mask])
        return obj_depth

    def get_3d_coordinates(self, u, v, disparity, img_w, img_h):
        # Approximate camera intrinsics
        # Assuming FOV ~ 60 degrees, focal length f ~ img_w
        f = img_w 
        
        # Disparity is proportional to 1/z
        # Avoid division by zero
        z = 1.0 / (disparity + 1e-2)
        
        x = (u - img_w / 2) * z / f
        y = (v - img_h / 2) * z / f
        
        return np.array([x, y, z])

    def analyze_video(self, video_path, num_frames=10):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if total_frames == 0:
            return None, None
            
        indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        frame_data = []
        prev_objects = []
        
        for i, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # 1. Detect Objects
            objects = self.sam.predict(image_pil)
            
            # 2. Estimate Depth
            depth_map = self.depth_model.predict(image_pil)
            
            # 3. Analyze Scene
            current_frame_objects = []
            
            for obj in objects:
                d = self.get_object_depth(obj, depth_map)
                box = obj["box"]
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                
                # Convert to 3D (d is disparity)
                pos_3d = self.get_3d_coordinates(cx, cy, d, width, height)
                
                obj_info = {
                    "id": None, # To be assigned
                    "label": obj["label"],
                    "disparity": float(d),
                    "center": (float(cx), float(cy)),
                    "pos_3d": pos_3d,
                    "box": box,
                    "velocity": 0.0, # Positive means approaching (distance decreasing)
                    "ttc": float('inf')
                }
                current_frame_objects.append(obj_info)
            
            # 4. Simple Tracking & Velocity Calculation
            if prev_objects:
                used_prev_indices = set()
                for curr_obj in current_frame_objects:
                    # Find closest object in prev frame with same label
                    best_match = None
                    best_prev_idx = -1
                    min_dist = float('inf')
                    
                    for p_idx, prev_obj in enumerate(prev_objects):
                        if p_idx in used_prev_indices: continue
                        if prev_obj["label"] == curr_obj["label"]:
                            # Use 2D distance for matching
                            dist = np.linalg.norm(np.array(curr_obj["center"]) - np.array(prev_obj["center"]))
                            if dist < min_dist:
                                min_dist = dist
                                best_match = prev_obj
                                best_prev_idx = p_idx
                    
                    # Threshold for matching (heuristic: 20% of image width)
                    if best_match and min_dist < width * 0.2: 
                        curr_obj["id"] = best_match["id"]
                        used_prev_indices.add(best_prev_idx)
                        
                        # Velocity: Decrease in distance (Z)
                        # prev_z - curr_z. If positive, object is getting closer.
                        # Units: relative depth units per frame
                        curr_obj["velocity"] = best_match["pos_3d"][2] - curr_obj["pos_3d"][2]
                        
                        # Calculate TTC (Time To Collision)
                        # TTC = Current Distance / Velocity
                        # If velocity is positive (approaching) and > 0
                        if curr_obj["velocity"] > 0.001:
                            curr_obj["ttc"] = curr_obj["pos_3d"][2] / curr_obj["velocity"]
                        else:
                            curr_obj["ttc"] = float('inf')
                    else:
                        curr_obj["id"] = str(idx) + "_" + str(len(current_frame_objects)) # New ID
                        curr_obj["ttc"] = float('inf')

            prev_objects = current_frame_objects
            
            # 5. Risk Assessment for this frame
            frame_risk_score = 0
            risk_reasons = []
            
            # A. Ego-Risk: Object approaching fast AND Low TTC
            for obj in current_frame_objects:
                # TTC Threshold: Relaxed to < 60 frames (approx 2-3 seconds)
                if obj["ttc"] < 60.0: 
                    frame_risk_score += 10
                    risk_reasons.append(f"CRITICAL: {obj['label']} collision course (TTC={obj['ttc']:.1f} frames)")
                elif obj["velocity"] > 0.02: # Lowered velocity threshold
                    frame_risk_score += 2
                    risk_reasons.append(f"{obj['label']} approaching fast")
                
                # Very close (Small Z) but check velocity too
                if obj["pos_3d"][2] < 0.2: # Increased distance threshold
                    if obj["velocity"] > -0.005: # Even slight movement or stationary close is risky
                        frame_risk_score += 5
                        risk_reasons.append(f"{obj['label']} critically close")
            
            # B. Interaction Risk: Objects close to each other
            for j, obj1 in enumerate(current_frame_objects):
                for k, obj2 in enumerate(current_frame_objects):
                    if j >= k: continue
                    
                    # Calculate 3D distance between objects
                    dist_objs = np.linalg.norm(obj1["pos_3d"] - obj2["pos_3d"])
                    
                    # Check for dangerous pairs
                    labels = {obj1["label"], obj2["label"]}
                    
                    vehicles = {"car", "truck", "bus"}
                    vulnerable = {"pedestrian", "motorcycle"}
                    
                    # Check if one is a vehicle and the other is vulnerable
                    if (labels & vehicles) and (labels & vulnerable):
                        # Threshold: relative distance increased
                        if dist_objs < 0.1: 
                            frame_risk_score += 5
                            risk_reasons.append(f"Proximity Warning: {obj1['label']} and {obj2['label']} close")
            
            frame_info = {
                "frame_idx": idx,
                "objects": current_frame_objects,
                "risk_score": frame_risk_score,
                "risk_reasons": risk_reasons,
                "image": image_pil
            }
            frame_data.append(frame_info)
            
        cap.release()
        
        # Select most dangerous frame
        if not frame_data:
            return None, None
            
        # Sort by risk_score (descending)
        dangerous_frame = max(frame_data, key=lambda x: x["risk_score"])
        
        return dangerous_frame, frame_data

    def generate_report(self, dangerous_frame, all_frames_data, output_image_path=None):
        self.load_vlm()
        
        # Visualize Bounding Boxes on the dangerous frame
        vis_image = np.array(dangerous_frame["image"]).copy()
        # Convert RGB to BGR for OpenCV
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        
        for obj in dangerous_frame['objects']:
            box = [int(c) for c in obj['box']]
            label = obj['label']
            color = (0, 255, 0) # Green
            
            # Highlight risky objects in Red
            if obj['ttc'] < 60.0 or obj['pos_3d'][2] < 0.2:
                color = (0, 0, 255) # Red
            
            cv2.rectangle(vis_image, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            # Add label and info
            info_text = f"{label} Z:{obj['pos_3d'][2]:.2f}"
            if obj['ttc'] < 1000:
                info_text += f" TTC:{obj['ttc']:.1f}"
                
            cv2.putText(vis_image, info_text, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if output_image_path:
            cv2.imwrite(output_image_path, vis_image)
            print(f"Saved visualization to {output_image_path}")
        
        structured_text = f"Analysis of Critical Frame {dangerous_frame['frame_idx']}:\n"
        structured_text += f"Risk Score: {dangerous_frame['risk_score']}\n"
        if dangerous_frame['risk_reasons']:
            structured_text += f"Detected Risks: {', '.join(dangerous_frame['risk_reasons'])}\n"
        
        structured_text += f"Detected {len(dangerous_frame['objects'])} objects:\n"
        for obj in dangerous_frame['objects']:
            # Velocity > 0 means approaching (distance decreasing)
            vel_str = "Approaching" if obj['velocity'] > 0.01 else ("Receding" if obj['velocity'] < -0.01 else "Stable")
            ttc_str = f"{obj['ttc']:.1f} frames" if obj['ttc'] < 1000 else "Safe"
            # Add Bounding Box info for debugging
            bbox_str = f"[{int(obj['box'][0])}, {int(obj['box'][1])}, {int(obj['box'][2])}, {int(obj['box'][3])}]"
            structured_text += f"- {obj['label']} | Box: {bbox_str} | Relative Distance (Z): {obj['pos_3d'][2]:.2f} | Motion: {vel_str} (Vel: {obj['velocity']:.3f}) | TTC: {ttc_str}\n"
            
        # Add context from previous frames (Trajectory)
        structured_text += "\nScene Context (Temporal):\n"
        # Find objects that have been tracked and are approaching
        approaching_objs = []
        for frame in all_frames_data:
            for obj in frame['objects']:
                if obj['velocity'] > 0.05:
                    approaching_objs.append(obj['label'])
        
        if approaching_objs:
            structured_text += f"Warning: The following objects showed rapid approach during the video: {set(approaching_objs)}.\n"

        # DEBUG: Print structured text to console to verify SAM3 output
        print(f"\n--- DEBUG: Structured Text for VLM ---\n{structured_text}\n--------------------------------------\n")

        prompt = f"""
        You are an expert traffic safety auditor. Your job is to identify ACTUAL accidents or POTENTIAL RISKS.
        
        Definitions:
        - Accident: A collision has occurred OR is about to occur.
        - High Risk: Vehicles/Pedestrians are dangerously close or on a collision course.
        - Safe: Clear road, no close interactions.
        
        Structured Data Analysis:
        {structured_text}
        
        Instructions:
        1. Analyze the "Detected Risks" and "TTC". 
           - If TTC < 60 frames, consider it HIGH RISK.
           - If objects are "critically close", consider it HIGH RISK.
        2. Look at the Image. 
           - Even if no crash is visible yet, if the situation looks dangerous, predict YES.
           - Better to be safe than sorry. High sensitivity is required.
        3. Think step-by-step:
           - Step 1: Describe the scene layout.
           - Step 2: Evaluate the motion of key objects.
           - Step 3: Assess the risk level (Low/Medium/High).
           - Step 4: Conclude.
        
        Output format:
        Accident Prediction: [Yes/No]
        Reasoning: [Your step-by-step analysis]
        """
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": dangerous_frame["image"]},
                    {"type": "text", "text": prompt},
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
        
        return output_text

def main():
    analyzer = AccidentAnalyzer()
    
    # Read dataset
    df = pd.read_csv("road_train_and_val_annot.csv")
    
    results = []
    
    # Process first few videos for demonstration
    for idx, row in tqdm(df.head(5).iterrows(), total=5):
        file_name = row['file_name']
        video_path = os.path.join(VIDEO_DIR, f"{file_name}.mp4")
        
        if not os.path.exists(video_path):
            print(f"Video {video_path} not found.")
            continue
            
        print(f"Processing {file_name}...")
        dangerous_frame, all_data = analyzer.analyze_video(video_path)
        
        if dangerous_frame:
            # Create output directory for visualizations
            vis_dir = "visualizations"
            os.makedirs(vis_dir, exist_ok=True)
            vis_path = os.path.join(vis_dir, f"{file_name}_frame_{dangerous_frame['frame_idx']}.jpg")
            
            report = analyzer.generate_report(dangerous_frame, all_data, output_image_path=vis_path)
            results.append({
                "file_name": file_name,
                "report": report,
                "dangerous_frame_idx": dangerous_frame["frame_idx"]
            })
            print(f"Report for {file_name}:\n{report}\n")
            torch.cuda.empty_cache()
        else:
            print(f"Could not analyze {file_name}")
            
    # Save results
    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f"Saved reports to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
