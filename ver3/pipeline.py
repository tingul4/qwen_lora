import os
# Restrict to a single GPU to prevent "device_map='auto'" from spreading across multiple cards
# and causing driver hangs/zombie processes.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import cv2
import gc
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
VIDEO_DIR = os.getenv("VIDEO_DIR", "/raid/mystery-project/dataset/road/train")
VIDEO_DIR = os.getenv("VIDEO_DIR", "/raid/mystery-project/dataset/road/test")
OUTPUT_FILE = "ver3_accident_report.csv"
HISTORY_LEN = 10 
FPS = 10 # Default FPS

# Models
YOLO_MODEL = "yolo11x.pt" 
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
QWEN_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

class Pipeline:
    def __init__(self):
        print(f"Initializing Pipeline on {DEVICE}...")
        
        self.detector = None
        self.depth_processor = None
        self.depth_model = None
        self.tracker = None
        self.vlm_model = None
        self.vlm_processor = None
        
        # State Tracking
        self.object_history = {}
        self.prev_gray = None # For Optical Flow
        self.bg_fast = None
        self.bg_slow = None

    def load_perception_models(self):
        if self.detector is None:
            print(f"Loading YOLOv11x ({YOLO_MODEL})...")
            self.detector = YOLO(YOLO_MODEL)
            
        if self.depth_model is None:
            print(f"Loading Depth Anything V2 ({DEPTH_MODEL_ID})...")
            self.depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).to(DEVICE)
            
        if self.tracker is None:
            print("Initializing ByteTrack...")
            # Increased thresholds to reduce false positives/jitter
            self.tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30)

    def unload_perception_models(self):
        print("Unloading perception models to free GPU memory...")
        if self.detector is not None:
            del self.detector
            self.detector = None
        if self.depth_model is not None:
            del self.depth_model
            self.depth_model = None
        if self.depth_processor is not None:
            del self.depth_processor
            self.depth_processor = None
        if self.tracker is not None:
            del self.tracker
            self.tracker = None
            
        gc.collect()
        torch.cuda.empty_cache()

    def load_vlm(self):
        if self.vlm_model is None:
            print("Loading Qwen2-VL (4-bit)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_enable_fp32_cpu_offload=True
            )
            self.vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
                QWEN_MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto"
            )
            self.vlm_processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID, min_pixels=256*28*28, max_pixels=1024*28*28)

    def unload_vlm(self):
        if self.vlm_model is not None:
            print("Unloading VLM...")
            del self.vlm_model
            self.vlm_model = None
        if self.vlm_processor is not None:
            del self.vlm_processor
            self.vlm_processor = None
        
        gc.collect()
        torch.cuda.empty_cache()

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
        h, w = depth_map.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        if x1 >= x2 or y1 >= y2:
            return 0.0
        
        # Use a central crop to ignore road/background in box
        cy_min = int(y1 + (y2-y1)*0.2)
        cy_max = int(y1 + (y2-y1)*0.8)
        cx_min = int(x1 + (x2-x1)*0.2)
        cx_max = int(x1 + (x2-x1)*0.8)
        
        if cx_min >= cx_max or cy_min >= cy_max:
             crop = depth_map[y1:y2, x1:x2] 
        else:
             crop = depth_map[cy_min:cy_max, cx_min:cx_max]

        return np.percentile(crop, 20)

    def get_world_state(self, box, depth, img_w):
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        norm_x = (center_x / img_w) - 0.5
        world_x = norm_x * depth * 1.5 
        world_z = depth
        return world_x, world_z

    def analyze_frames(self, frames_dir):
        # We assume perception models are loaded by the caller in Batch Mode
        # But for safety, check here
        if self.detector is None:
            self.load_perception_models()

        if torch.cuda.is_available():
            free_mem, _ = torch.cuda.mem_get_info()
            if free_mem > 10 * 1024**3:
                sample_rate = 1
                print(f"Memory sufficient. Processing ALL frames.")
            else:
                sample_rate = 5
                print(f"Memory tight. Processing every 5th frame.")
        else:
            sample_rate = 5

        try:
            image_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        except FileNotFoundError:
            return None

        if not image_files:
            return None

        self.object_history = {} 
        self.prev_gray = None 
        
        # === 重置背景 ===
        self.bg_fast = None
        self.bg_slow = None
        
        # Proper Reset of Tracker
        self.tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30)
        
        last_frame_package = None
        
        for frame_idx, img_file in enumerate(image_files):
            if frame_idx % sample_rate != 0:
                continue
                
            img_path = os.path.join(frames_dir, img_file)
            frame = cv2.imread(img_path)
            if frame is None: continue
            
            # Prepare Grayscale for Optical Flow
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            
            # ========== 新增：背景模型更新 ==========
            if self.bg_fast is None:
                self.bg_fast = curr_gray.copy()
                self.bg_slow = curr_gray.copy()
                recently_stopped = np.zeros_like(curr_gray, dtype=bool)
            else:
                self.bg_fast = cv2.addWeighted(curr_gray, 0.3, self.bg_fast, 0.7, 0)
                if frame_idx % 5 == 0:
                    self.bg_slow = cv2.addWeighted(curr_gray, 0.05, self.bg_slow, 0.95, 0)
            
            fg_fast = cv2.absdiff(curr_gray, self.bg_fast) > 20
            fg_slow = cv2.absdiff(curr_gray, self.bg_slow) > 20
            recently_stopped = fg_fast & ~fg_slow
            # =====================================
            
            # 1. Detect & Track
            # Increased confidence to 0.5 to reduce jitter/false positives
            # Classes: 2=Car, 3=Motorcycle, 5=Bus, 7=Truck (COCO)
            results = self.detector(image_rgb, verbose=False, conf=0.5, classes=[2, 3, 5, 7])[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = self.tracker.update_with_detections(detections)
            
            # 2. Depth
            depth_map = self.estimate_depth(image_pil)
            img_h, img_w = depth_map.shape
            
            # --- EGO MOTION ESTIMATION ---
            # Calculate global flow (camera movement) by tracking background
            ego_flow = np.array([0.0, 0.0])
            
            if self.prev_gray is not None:
                # 1. Create a mask that IGNORES all detected objects
                bg_mask = np.ones_like(self.prev_gray) * 255
                
                # We need bounding boxes from the PREVIOUS frame to mask correctly, 
                # but using current frame detections is a decent approximation for fast motion
                # Ideally we track detection history, but let's mask current detections on prev frame
                for box in detections.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    # Dilate box slightly to ensure object edges are masked
                    x1 = max(0, x1 - 10)
                    y1 = max(0, y1 - 10)
                    x2 = min(img_w, x2 + 10)
                    y2 = min(img_h, y2 + 10)
                    bg_mask[y1:y2, x1:x2] = 0
                
                # 2. Track background features
                p0_bg = cv2.goodFeaturesToTrack(self.prev_gray, mask=bg_mask, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
                
                if p0_bg is not None:
                    p1_bg, st_bg, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, p0_bg, None, winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                    
                    if p1_bg is not None:
                        good_new_bg = p1_bg[st_bg==1]
                        good_old_bg = p0_bg[st_bg==1]
                        
                        if len(good_new_bg) > 0:
                            flow_bg = good_new_bg - good_old_bg
                            # Median flow of background = Ego Motion
                            ego_flow = np.median(flow_bg, axis=0)

            # 3. Process Objects & Kinematics with Optical Flow
            frame_objects = []
            
            for i, (box, _, _, class_id, tracker_id, _) in enumerate(detections):
                if tracker_id is None: continue
                track_id = int(tracker_id)
                class_name = results.names[int(class_id)]
                
                # 3D State
                z_depth = self.get_object_depth(box, depth_map)
                x_lat, z_long = self.get_world_state(box, z_depth, img_w)
                
                # 2D Box
                x1, y1, x2, y2 = map(int, box)
                box_w = x2 - x1
                box_h = y2 - y1
                
                # --- OPTICAL FLOW CALCULATION ---
                # Default velocities
                v_pix_x = 0.0
                v_pix_y = 0.0
                
                if track_id in self.object_history and self.prev_gray is not None:
                    prev_state = self.object_history[track_id][-1]
                    px1, py1, px2, py2 = map(int, prev_state['box'])
                    
                    # 1. Mask for the object: SHRINK ROI to avoid background
                    # Shrink by 20% on each side
                    roi_mask = np.zeros_like(self.prev_gray)
                    sx1 = int(px1 + box_w * 0.2)
                    sy1 = int(py1 + box_h * 0.2)
                    sx2 = int(px2 - box_w * 0.2)
                    sy2 = int(py2 - box_h * 0.2)
                    
                    if sx2 > sx1 and sy2 > sy1:
                        roi_mask[sy1:sy2, sx1:sx2] = 255
                        
                        p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=roi_mask, maxCorners=20, qualityLevel=0.3, minDistance=7, blockSize=7)
                        
                        if p0 is not None:
                            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, p0, None, winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                            if p1 is not None:
                                good_new = p1[st==1]
                                good_old = p0[st==1]
                                if len(good_new) > 0:
                                    flows = good_new - good_old
                                    raw_obj_flow = np.median(flows, axis=0)
                                    
                                    # COMPENSATE FOR EGO MOTION
                                    net_flow = raw_obj_flow - ego_flow
                                    
                                    v_pix_x = net_flow[0] * FPS
                                    v_pix_y = net_flow[1] * FPS
                
                if track_id not in self.object_history:
                    self.object_history[track_id] = deque(maxlen=HISTORY_LEN)
                
                current_state = {
                    'frame': frame_idx, 'box': box,
                    'x': x_lat, 'z': z_long, 
                    'v_pix_x': v_pix_x, 'v_pix_y': v_pix_y,
                    'box_center': ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                }
                self.object_history[track_id].append(current_state)
                
                vx = 0.0
                vz = 0.0
                # Box velocity (pixel space) for IoU-based prediction
                box_vx = 0.0
                box_vy = 0.0
                
                history = self.object_history[track_id]
                if len(history) >= 2:
                    recent = history[-1]
                    past = history[-2] 
                    dt = (recent['frame'] - past['frame']) / FPS
                    if dt > 0:
                        vz = (recent['z'] - past['z']) / dt
                        # Use Compensated Optical Flow for Lateral Velocity
                        # Increased coefficient for better lateral velocity estimation
                        vx = v_pix_x * 0.005 * recent['z'] 
                        
                        # Calculate box center velocity in pixel space
                        prev_box = past['box']
                        prev_cx = (prev_box[0] + prev_box[2]) / 2
                        prev_cy = (prev_box[1] + prev_box[3]) / 2
                        curr_cx = (box[0] + box[2]) / 2
                        curr_cy = (box[1] + box[3]) / 2
                        box_vx = (curr_cx - prev_cx) / dt
                        box_vy = (curr_cy - prev_cy) / dt

                # ========== IMPROVED MOTION STATE DETECTION ==========
                # Calculate total motion magnitude
                total_box_speed = np.sqrt(box_vx**2 + box_vy**2)
                total_flow_speed = np.sqrt(v_pix_x**2 + v_pix_y**2)
                
                # === 方案A：背景模型靜止檢測 ===
                is_stationary_bg = False
                if frame_idx > 0:
                    try:
                        roi_recently_stopped = recently_stopped[int(y1):int(y2), int(x1):int(x2)]
                        if roi_recently_stopped.size > 0:
                            ratio_stopped = roi_recently_stopped.sum() / roi_recently_stopped.size
                            if ratio_stopped > 0.3:
                                is_stationary_bg = True
                    except:
                        pass
                
                # === 方案B：幀差分靜止檢測 ===
                is_stationary_diff = False
                if self.prev_gray is not None:
                    try:
                        frame_diff = cv2.absdiff(self.prev_gray, curr_gray)
                        roi_diff = frame_diff[int(y1):int(y2), int(x1):int(x2)]
                        if roi_diff.size > 0:
                            change_ratio = (roi_diff > 15).sum() / roi_diff.size
                            if change_ratio < 0.01:
                                is_stationary_diff = True
                    except:
                        pass
                
                # --- 1. STATIONARY DETECTION ---
                # Object is stationary if box barely moves AND optical flow is minimal
                box_move_threshold = img_w * 0.005
                flow_threshold = img_w * 0.01
                is_stationary_motion = (total_box_speed < box_move_threshold) and (total_flow_speed < flow_threshold)
                
                # 合併三種方法
                is_stationary = is_stationary_bg or is_stationary_diff or is_stationary_motion
                
                # --- 2. LATERAL MOVEMENT DETECTION ---
                # Must have significant lateral flow relative to image width
                lateral_threshold = img_w * 0.02  # 2% of image width
                is_lateral_move = abs(v_pix_x) > lateral_threshold and not is_stationary
                
                # === 靜止物體沒有側向移動 ===
                if is_stationary:
                    is_lateral_move = False
                
                # --- 3. TURNING DETECTION with multiple criteria ---
                is_turning = False
                turning_confidence = 0.0
                
                # === 靜止物體不能轉彎 ===
                if is_stationary:
                    is_turning = False
                    turning_confidence = 0.0
                elif not is_stationary and is_lateral_move:
                    # Criterion A: Lateral flow dominates vertical flow
                    if abs(v_pix_y) > 0.1:
                        lateral_ratio = abs(v_pix_x) / (abs(v_pix_y) + 1e-6)
                    else:
                        lateral_ratio = abs(v_pix_x) / (flow_threshold + 1e-6)
                    
                    if lateral_ratio > 1.2:  # Lateral is 20% stronger than vertical
                        turning_confidence += 0.3
                    
                    # Criterion B: Check trajectory consistency over multiple frames
                    if len(history) >= 3:
                        # Get lateral velocities from recent frames
                        recent_lat_flows = [h.get('v_pix_x', 0) for h in list(history)[-3:]]
                        
                        # Check if lateral movement is consistent (same direction)
                        signs = [np.sign(f) for f in recent_lat_flows if abs(f) > lateral_threshold * 0.5]
                        if len(signs) >= 2:
                            # All non-zero movements in same direction
                            if all(s == signs[0] for s in signs) and signs[0] != 0:
                                turning_confidence += 0.3
                            
                            # Calculate average lateral movement
                            avg_lat = np.mean([abs(f) for f in recent_lat_flows])
                            if avg_lat > lateral_threshold:
                                turning_confidence += 0.2
                    
                    # Criterion C: Box center trajectory analysis
                    if len(history) >= 3:
                        centers = [h.get('box_center', (0, 0)) for h in list(history)[-3:]]
                        if all(c[0] != 0 for c in centers):
                            # Calculate trajectory direction changes
                            dx1 = centers[1][0] - centers[0][0]
                            dx2 = centers[2][0] - centers[1][0]
                            
                            # Consistent lateral movement in same direction
                            if np.sign(dx1) == np.sign(dx2) and abs(dx1) > 3 and abs(dx2) > 3:
                                turning_confidence += 0.2
                    
                    # Final turning decision
                    is_turning = turning_confidence >= 0.5
                
                # === 多幀穩定性檢查 ===
                if len(history) >= 5:
                    stationary_count = sum(1 for h in list(history)[-5:] if h.get('is_stationary', False))
                    if stationary_count >= 2:
                        is_turning = False
                        turning_confidence = min(turning_confidence, 0.2)
                
                # --- 4. STRAIGHT MOVEMENT CHECK ---
                # If vertical movement strongly dominates, it's likely straight motion
                if not is_stationary and abs(v_pix_y) > abs(v_pix_x) * 2.0:
                    is_turning = False
                    is_lateral_move = False

                obj_data = {
                    "id": track_id,
                    "class": class_name,
                    "box": box,
                    "pos": (x_lat, z_long), 
                    "vel": (vx, vz),       
                    "depth": z_depth,
                    "is_turning": is_turning,
                    "is_lateral": is_lateral_move,
                    "is_stationary": is_stationary,
                    "turning_confidence": turning_confidence if not is_stationary else 0.0,
                    "flow_vel": (v_pix_x, v_pix_y),
                    "box_vel": (box_vx, box_vy),  # Pixel-space box velocity
                    "box_size": (box_w, box_h),    # Box dimensions for IoU calculation
                    "total_speed": total_box_speed
                }
                frame_objects.append(obj_data)

            self.prev_gray = curr_gray.copy()
            
            # --- LAST FRAME PREDICTION LOGIC ---
            current_collisions = []
            TIME_HORIZON = 1.5  # Extended to 1.5 seconds for turning scenarios
            STEPS = int(TIME_HORIZON / 0.1)
            
            def calc_iou(box1, box2):
                """Calculate IoU between two boxes [x1, y1, x2, y2]"""
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])
                
                inter_w = max(0, x2 - x1)
                inter_h = max(0, y2 - y1)
                inter_area = inter_w * inter_h
                
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                union_area = area1 + area2 - inter_area
                
                return inter_area / (union_area + 1e-6)
            
            def predict_box(obj, t):
                """Predict box position at time t based on box velocity"""
                x1, y1, x2, y2 = obj['box']
                vx, vy = obj.get('box_vel', (0, 0))
                dx = vx * t
                dy = vy * t
                return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
            
            def boxes_approaching(obj_a, obj_b):
                """Check if two objects are approaching each other"""
                # Current centers
                cx_a = (obj_a['box'][0] + obj_a['box'][2]) / 2
                cy_a = (obj_a['box'][1] + obj_a['box'][3]) / 2
                cx_b = (obj_b['box'][0] + obj_b['box'][2]) / 2
                cy_b = (obj_b['box'][1] + obj_b['box'][3]) / 2
                
                # Vector from A to B
                dx = cx_b - cx_a
                dy = cy_b - cy_a
                dist = np.sqrt(dx*dx + dy*dy) + 1e-6
                
                # Relative velocity (B relative to A)
                vx_a, vy_a = obj_a.get('box_vel', (0, 0))
                vx_b, vy_b = obj_b.get('box_vel', (0, 0))
                rel_vx = vx_b - vx_a
                rel_vy = vy_b - vy_a
                
                # Dot product: negative means approaching
                approach_rate = (dx * rel_vx + dy * rel_vy) / dist
                return approach_rate < -5  # Approaching at > 5 pixels/sec
            
            for i in range(len(frame_objects)):
                for j in range(i + 1, len(frame_objects)):
                    obj_a = frame_objects[i]
                    obj_b = frame_objects[j]
                    
                    # --- METHOD 1: 3D World Space Prediction ---
                    dist_sq = (obj_a['pos'][0]-obj_b['pos'][0])**2 + (obj_a['pos'][1]-obj_b['pos'][1])**2
                    
                    # Relaxed distance filter for turning vehicles
                    max_dist_sq = 900  # 30m
                    if obj_a['is_turning'] or obj_b['is_turning']:
                        max_dist_sq = 1600  # 40m for turning scenarios
                    
                    min_dist_3d = float('inf')
                    time_at_min_dist_3d = 0
                    impact_v_rel = 0
                    
                    if dist_sq <= max_dist_sq:
                        for step in range(1, STEPS + 1):
                            t = step * 0.1
                            ax = obj_a['pos'][0] + obj_a['vel'][0] * t
                            az = obj_a['pos'][1] + obj_a['vel'][1] * t
                            bx = obj_b['pos'][0] + obj_b['vel'][0] * t
                            bz = obj_b['pos'][1] + obj_b['vel'][1] * t
                            
                            dist = np.sqrt((ax - bx)**2 + (az - bz)**2)
                            
                            if dist < min_dist_3d:
                                min_dist_3d = dist
                                time_at_min_dist_3d = t
                                avx, avz = obj_a['vel']
                                bvx, bvz = obj_b['vel']
                                impact_v_rel = np.sqrt((avx-bvx)**2 + (avz-bvz)**2)
                    
                    # --- METHOD 2: 2D Box IoU Prediction ---
                    max_iou = 0.0
                    time_at_max_iou = 0
                    
                    for step in range(1, STEPS + 1):
                        t = step * 0.1
                        pred_box_a = predict_box(obj_a, t)
                        pred_box_b = predict_box(obj_b, t)
                        iou = calc_iou(pred_box_a, pred_box_b)
                        
                        if iou > max_iou:
                            max_iou = iou
                            time_at_max_iou = t
                    
                    # --- METHOD 3: Current proximity + approaching detection ---
                    current_iou = calc_iou(obj_a['box'], obj_b['box'])
                    is_approaching = boxes_approaching(obj_a, obj_b)
                    
                    # Check if both objects are stationary (no collision possible between parked cars)
                    both_stationary = obj_a.get('is_stationary', False) and obj_b.get('is_stationary', False)
                    
                    # Adaptive collision radius based on depth uncertainty
                    avg_depth = (obj_a['depth'] + obj_b['depth']) / 2
                    collision_radius = 3.0 + avg_depth * 0.1  # Larger radius for farther objects
                    
                    # Only increase radius for CONFIRMED turning (high confidence)
                    a_turning_conf = obj_a.get('turning_confidence', 0)
                    b_turning_conf = obj_b.get('turning_confidence', 0)
                    if a_turning_conf >= 0.5 or b_turning_conf >= 0.5:
                        collision_radius += 1.5
                    elif obj_a.get('is_lateral', False) or obj_b.get('is_lateral', False):
                        collision_radius += 0.8
                    
                    # --- PROBABILITY CALCULATION (0-100%) ---
                    prob = 0.0
                    reason = "Safe"
                    best_time = 0
                    
                    # Skip if both objects are stationary
                    if both_stationary:
                        prob = 0.0
                        reason = "Both Stationary"
                    else:
                        # Score from 3D prediction
                        if min_dist_3d < collision_radius:
                            prob += 40.0
                            prob += (1.0 / (time_at_min_dist_3d + 0.5)) * 10.0
                            prob += (1.0 / (min_dist_3d + 0.5)) * 10.0
                            prob += impact_v_rel * 1.5
                            best_time = time_at_min_dist_3d
                            reason = "Collision Predicted (3D)"
                        elif min_dist_3d < (collision_radius + 3.0):
                            prob += 15.0
                            prob += (1.0 / (min_dist_3d - collision_radius + 0.5)) * 5.0
                            best_time = time_at_min_dist_3d
                        
                        # Score from 2D IoU prediction
                        if max_iou > 0.01:  # Any predicted overlap
                            iou_bonus = min(30.0, max_iou * 150)  # Up to 30% bonus
                            prob += iou_bonus
                            if max_iou > 0.1:
                                reason = "Box Overlap Predicted"
                                best_time = time_at_max_iou
                        
                        # Score from current state
                        if current_iou > 0:
                            prob += 25.0  # Already overlapping!
                            reason = "Current Overlap Detected"
                            best_time = 0
                        
                        # Approaching bonus (only if at least one object is moving)
                        if is_approaching:
                            prob += 10.0
                        
                        # Turning/lateral bonuses - use confidence-weighted scoring
                        # High confidence turning gets full bonus
                        if a_turning_conf >= 0.5 or b_turning_conf >= 0.5:
                            turn_bonus = 15.0 * max(a_turning_conf, b_turning_conf)
                            prob += turn_bonus
                            if "Collision" in reason or "Overlap" in reason:
                                reason = "Turning Collision"
                        # Low confidence lateral movement gets reduced bonus
                        elif obj_a.get('is_lateral', False) or obj_b.get('is_lateral', False):
                            # Only add bonus if at least one object is actually moving
                            if not obj_a.get('is_stationary', False) or not obj_b.get('is_stationary', False):
                                prob += 8.0
                                if "Collision" in reason or "Overlap" in reason:
                                    reason = "Cut-in Collision"
                        
                        # Penalty for stationary objects (reduce false positives from parked cars)
                        if obj_a.get('is_stationary', False) or obj_b.get('is_stationary', False):
                            prob *= 0.7  # 30% reduction if one object is stationary
                    
                    prob = min(99.9, prob)
                    
                    if prob > 25.0:  # Lowered threshold
                        collision_info = {
                            "obj1": obj_a, "obj2": obj_b,
                            "prob": prob, "t_cpa": best_time,
                            "pred_dist": min_dist_3d, 
                            "pred_iou": max_iou,
                            "reason": reason
                        }
                        current_collisions.append(collision_info)

            scene_desc = {
                "frame_id": frame_idx,
                "collisions": current_collisions,
                "all_objects": frame_objects
            }
            
            last_frame_package = {
                "frame_idx": frame_idx,
                "image": image_pil,
                "json_desc": scene_desc
            }

        return last_frame_package

    def generate_report_prompt(self, scene_json):
        desc = f"Analysis of the Final Frame (Frame {scene_json['frame_id']}):\n"
        
        if not scene_json['collisions']:
             desc += "Prediction: No immediate collision predicted in the next 1 second (Probability < 30%).\n"
        else:
             scene_json['collisions'].sort(key=lambda x: x['prob'], reverse=True)
             top_risk = scene_json['collisions'][0]
             o1 = top_risk['obj1']
             o2 = top_risk['obj2']
             
             desc += f"PREDICTION RESULT:\n"
             desc += f"Crash Probability: {top_risk['prob']:.1f}%\n"
             desc += f"Potential Scenario: {top_risk['reason']} between {o1['class']} (ID {o1['id']}) and {o2['class']} (ID {o2['id']}).\n"
             desc += f"Time to Impact: {top_risk['t_cpa']:.1f} seconds.\n"
        
        desc += "\nTask: Based on the visual evidence and this prediction, describe the potential accident scenario and the estimated likelihood of a crash."
        return desc

    def run_vlm_inference(self, frame_data):
        if self.vlm_model is None:
            self.unload_perception_models()
            self.load_vlm()
        
        prompt_text = self.generate_report_prompt(frame_data['json_desc'])
        
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
    critical_candidates = []
    
    print("\n=== PHASE 1: PERCEPTION ANALYSIS (YOLO + Depth + Optical Flow) ===")
    
    pipeline.unload_vlm()
    pipeline.load_perception_models()
    
    for idx, row in tqdm(df[180:185].iterrows(), total=5):
        file_name = row['file_name']
        frames_dir = os.path.join(VIDEO_DIR, file_name)
        
        if not os.path.exists(frames_dir):
            print(f"Frames directory {frames_dir} not found.")
            continue
            
        print(f"Processing {file_name}...")
        final_frame_pkg = pipeline.analyze_frames(frames_dir)
        
        if final_frame_pkg:
            risk_found = len(final_frame_pkg['json_desc']['collisions']) > 0
            if risk_found:
                print(f"  -> Prediction found in Final Frame {final_frame_pkg['frame_idx']}")
            
            critical_candidates.append({
                "file_name": file_name,
                "data": final_frame_pkg
            })

    if not critical_candidates:
        print("No video frames processed. Exiting.")
        return
    
    print("\n=== PHASE 2: RISK REPORT GENERATION (Qwen2-VL) ===")
    pipeline.unload_perception_models()
    
    for item in tqdm(critical_candidates, desc="Generating Reports"):
        file_name = item['file_name']
        frame_data = item['data']
        
        print(f"Generating report for {file_name}...")
        report, prompt = pipeline.run_vlm_inference(frame_data)
        
        vis_dir = "ver3_visualizations"
        os.makedirs(vis_dir, exist_ok=True)
        vis_path = os.path.join(vis_dir, f"{file_name}_critical.jpg")
        
        img_cv = cv2.cvtColor(np.array(frame_data['image']), cv2.COLOR_RGB2BGR)
        
        if 'all_objects' in frame_data['json_desc']:
            for obj in frame_data['json_desc']['all_objects']:
                x1, y1, x2, y2 = map(int, obj['box'])
                
                # Color code based on state
                if obj.get('is_stationary', False):
                    box_color = (128, 128, 128)  # Gray for stationary
                elif obj.get('is_turning', False):
                    box_color = (0, 165, 255)  # Orange for turning
                elif obj.get('is_lateral', False):
                    box_color = (0, 255, 255)  # Yellow for lateral
                else:
                    box_color = (255, 255, 255)  # White for moving straight
                
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), box_color, 2)
                
                if 'flow_vel' in obj:
                    cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                    fv_x, fv_y = obj['flow_vel']
                    end_x = int(cx + fv_x * 0.5) 
                    end_y = int(cy + fv_y * 0.5)
                    cv2.arrowedLine(img_cv, (cx, cy), (end_x, end_y), (0, 255, 0), 2)
                
                # Status label with confidence
                status = []
                if obj.get('is_stationary', False):
                    status.append("STOP")
                elif obj.get('is_turning', False):
                    conf = obj.get('turning_confidence', 0)
                    status.append(f"TURN({conf:.1f})")
                elif obj.get('is_lateral', False):
                    status.append("LAT")
                
                if status:
                    cv2.putText(img_cv, " ".join(status), (x1, y1-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        if 'collisions' in frame_data['json_desc']:
            for c in frame_data['json_desc']['collisions']:
                o1 = c['obj1']
                o2 = c['obj2']
                c1_x = int((o1['box'][0]+o1['box'][2])/2)
                c1_y = int((o1['box'][1]+o1['box'][3])/2)
                c2_x = int((o2['box'][0]+o2['box'][2])/2)
                c2_y = int((o2['box'][1]+o2['box'][3])/2)
                cv2.line(img_cv, (c1_x, c1_y), (c2_x, c2_y), (0, 0, 255), 4)
                
                prob_label = f"{c['prob']:.0f}%"
                label_x = int((c1_x + c2_x)/2)
                label_y = int((c1_y + c2_y)/2)
                cv2.putText(img_cv, prob_label, (label_x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                       
        cv2.imwrite(vis_path, img_cv)
        
        results.append({
            "file_name": file_name,
            "prompt": prompt,
            "report": report,
            "frame_id": frame_data['frame_idx']
        })
        print(f"Report:\n{report}\n")

    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f"Saved results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
