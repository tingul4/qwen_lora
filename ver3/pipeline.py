import os
# Restrict to a single GPU to prevent "device_map='auto'" from spreading across multiple cards
# and causing driver hangs/zombie processes.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import cv2
import gc
import numpy as np
import pandas as pd
import torch
import logging
from datetime import datetime
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "/raid/mystery-project/dataset/road/train"
TEST_DIR = "/raid/mystery-project/dataset/road/test"
ANNOT_FILE = "/raid/mystery-project/qwen_lora/ver2/road_train_and_val_annot.csv"
OUTPUT_FILE = "ver3_accident_report.csv"
HISTORY_LEN = 10 
FPS = 10 
EVAL_TEST = True # Set to True to skip train data

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
        self.object_history = {}
        self.prev_gray = None
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
            self.tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30)

    def unload_perception_models(self):
        if self.detector is not None: del self.detector; self.detector = None
        if self.depth_model is not None: del self.depth_model; self.depth_model = None
        if self.depth_processor is not None: del self.depth_processor; self.depth_processor = None
        if self.tracker is not None: del self.tracker; self.tracker = None
        gc.collect()
        torch.cuda.empty_cache()

    def unload_vlm(self):
        if self.vlm_model is not None: del self.vlm_model; self.vlm_model = None
        if self.vlm_processor is not None: del self.vlm_processor; self.vlm_processor = None
        gc.collect()
        torch.cuda.empty_cache()

    def estimate_depth(self, image_pil):
        inputs = self.depth_processor(images=image_pil, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth
        prediction = torch.nn.functional.interpolate(predicted_depth.unsqueeze(1), size=image_pil.size[::-1], mode="bicubic", align_corners=False)
        return prediction.squeeze().cpu().numpy()

    def get_object_depth(self, box, depth_map):
        x1, y1, x2, y2 = map(int, box)
        h, w = depth_map.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x1 >= x2 or y1 >= y2: return 0.0
        cy_min, cy_max = int(y1 + (y2-y1)*0.2), int(y1 + (y2-y1)*0.8)
        cx_min, cx_max = int(x1 + (x2-x1)*0.2), int(x1 + (x2-x1)*0.8)
        crop = depth_map[cy_min:cy_max, cx_min:cx_max] if cx_min < cx_max and cy_min < cy_max else depth_map[y1:y2, x1:x2]
        return np.percentile(crop, 20)

    def get_world_state(self, box, depth, img_w):
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        norm_x = (center_x / img_w) - 0.5
        world_x = norm_x * depth * 1.5 
        world_z = depth
        return world_x, world_z

    def analyze_frames(self, frames_dir):
        if self.detector is None: self.load_perception_models()
        try:
            image_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        except Exception: return None
        if not image_files: return None

        self.object_history = {} 
        self.prev_gray = None 
        self.bg_fast = None
        self.bg_slow = None
        self.tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30)
        
        last_frame_package = None
        sample_rate = 1

        for frame_idx, img_file in enumerate(image_files):
            img_path = os.path.join(frames_dir, img_file)
            frame = cv2.imread(img_path)
            if frame is None: continue
            
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_h, img_w = curr_gray.shape
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            
            # Background Modeling
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
            
            results = self.detector(frame, verbose=False, conf=0.5, classes=[2, 3, 5, 7])[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = self.tracker.update_with_detections(detections)
            
            # Ego Motion
            ego_flow = np.array([0.0, 0.0])
            bg_jitter = 0.0
            if self.prev_gray is not None:
                bg_mask = np.ones_like(self.prev_gray) * 255
                for box in detections.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    bg_mask[max(0,y1-10):min(img_h,y2+10), max(0,x1-10):min(img_w,x2+10)] = 0
                p0_bg = cv2.goodFeaturesToTrack(self.prev_gray, mask=bg_mask, maxCorners=50, qualityLevel=0.3, minDistance=7)
                if p0_bg is not None:
                    p1_bg, st_bg, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, p0_bg, None)
                    if p1_bg is not None and len(p1_bg[st_bg==1]) > 0:
                        flow_bg = p1_bg[st_bg==1] - p0_bg[st_bg==1]
                        ego_flow = np.median(flow_bg, axis=0)
                        bg_jitter = np.std(flow_bg)

            depth_map = self.estimate_depth(image_pil)
            frame_objects = []
            
            for i, (box, _, _, class_id, tracker_id, _) in enumerate(detections):
                if tracker_id is None: continue
                tid = int(tracker_id)
                x1, y1, x2, y2 = map(int, box)
                
                z_depth = self.get_object_depth(box, depth_map)
                x_lat, z_long = self.get_world_state(box, z_depth, img_w)
                v_pix_x, v_pix_y = 0.0, 0.0
                
                if tid in self.object_history and self.prev_gray is not None:
                    prev_state = self.object_history[tid][-1]
                    px1, py1, px2, py2 = map(int, prev_state['box'])
                    roi_mask = np.zeros_like(self.prev_gray)
                    roi_mask[int(py1+(py2-py1)*0.2):int(py2-(py2-py1)*0.2), int(px1+(px2-px1)*0.2):int(px2-(px2-px1)*0.2)] = 255
                    p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=roi_mask, maxCorners=10, qualityLevel=0.3, minDistance=7)
                    if p0 is not None:
                        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, p0, None)
                        if p1 is not None and len(p1[st==1]) > 0:
                            raw_flow = np.median(p1[st==1] - p0[st==1], axis=0)
                            net_flow = raw_flow - ego_flow
                            v_pix_x, v_pix_y = net_flow[0] * FPS, net_flow[1] * FPS
                
                if tid not in self.object_history: self.object_history[tid] = deque(maxlen=HISTORY_LEN)
                self.object_history[tid].append({'frame': frame_idx, 'box': box, 'x': x_lat, 'z': z_long, 'v_pix_x': v_pix_x})
                
                vx, vz, box_vx, box_vy = 0.0, 0.0, 0.0, 0.0
                if len(self.object_history[tid]) >= 2:
                    h = self.object_history[tid]
                    dt = (h[-1]['frame'] - h[-2]['frame']) / FPS
                    if dt > 0:
                        vz = (h[-1]['z'] - h[-2]['z']) / dt
                        vx = v_pix_x * 0.005 * h[-1]['z']
                        pb, cb = h[-2]['box'], box
                        box_vx = ((cb[0]+cb[2])/2 - (pb[0]+pb[2])/2) / dt
                        box_vy = ((cb[1]+cb[3])/2 - (pb[1]+pb[3])/2) / dt

                # State Detection
                total_box_speed = np.sqrt(box_vx**2 + box_vy**2)
                total_flow_speed = np.sqrt(v_pix_x**2 + v_pix_y**2)
                
                is_stat_bg, is_stat_diff = False, False
                if frame_idx > 0:
                    roi_stop = recently_stopped[int(y1):int(y2), int(x1):int(x2)]
                    if roi_stop.size > 0 and (roi_stop.sum()/roi_stop.size) > 0.3: is_stat_bg = True
                if self.prev_gray is not None:
                    roi_d = cv2.absdiff(self.prev_gray, curr_gray)[int(y1):int(y2), int(x1):int(x2)]
                    if roi_d.size > 0 and ((roi_d > 15).sum()/roi_d.size) < 0.01: is_stat_diff = True
                
                is_stationary = is_stat_bg or is_stat_diff or (total_box_speed < img_w*0.005 and total_flow_speed < max(img_w*0.01, bg_jitter*3.0))
                is_lat = abs(v_pix_x) > max(img_w*0.02, bg_jitter*4.0) and not is_stationary
                
                is_turn = False
                if is_lat:
                    ratio = abs(v_pix_x)/(abs(v_pix_y)+1e-6)
                    if ratio > 1.2: is_turn = True
                
                if not is_stationary and abs(v_pix_y) > abs(v_pix_x)*2.0: is_turn = False; is_lat = False

                frame_objects.append({
                    "id": tid, "box": box, "pos": (x_lat, z_long), "vel": (vx, vz), "depth": z_depth,
                    "is_turning": is_turn, "is_lateral": is_lat, "is_stationary": is_stationary, 
                    "flow_vel": (v_pix_x, v_pix_y), "box_vel": (box_vx, box_vy)
                })

            self.prev_gray = curr_gray.copy()
            
            # Prediction
            current_collisions = []
            for i in range(len(frame_objects)):
                for j in range(i + 1, len(frame_objects)):
                    oa, ob = frame_objects[i], frame_objects[j]
                    if (oa['pos'][0]-ob['pos'][0])**2 + (oa['pos'][1]-ob['pos'][1])**2 > 900: continue
                    
                    min_dist, time_at_min, impact_v = float('inf'), 0, 0
                    for step in range(1, 16): # 1.5s
                        t = step * 0.1
                        ax, az = oa['pos'][0] + oa['vel'][0]*t,oa['pos'][1] + oa['vel'][1]*t
                        bx, bz = ob['pos'][0] + ob['vel'][0]*t,ob['pos'][1] + ob['vel'][1]*t
                        dist = np.sqrt((ax-bx)**2 + (az-bz)**2)
                        if dist < min_dist: min_dist = dist; time_at_min = t; impact_v = np.sqrt((oa['vel'][0]-ob['vel'][0])**2 + (oa['vel'][1]-ob['vel'][1])**2)
                    
                    # 2D IoU Prediction
                    max_iou = 0.0
                    for step in range(1, 16):
                        t = step * 0.1
                        pba = [oa['box'][0]+oa['box_vel'][0]*t, oa['box'][1]+oa['box_vel'][1]*t, oa['box'][2]+oa['box_vel'][0]*t, oa['box'][3]+oa['box_vel'][1]*t]
                        pbb = [ob['box'][0]+ob['box_vel'][0]*t, ob['box'][1]+ob['box_vel'][1]*t, ob['box'][2]+ob['box_vel'][0]*t, ob['box'][3]+ob['box_vel'][1]*t]
                        
                        # IoU Calc
                        xA = max(pba[0], pbb[0]); yA = max(pba[1], pbb[1])
                        xB = min(pba[2], pbb[2]); yB = min(pba[3], pbb[3])
                        inter = max(0, xB - xA) * max(0, yB - yA)
                        areaA = (pba[2] - pba[0]) * (pba[3] - pba[1])
                        areaB = (pbb[2] - pbb[0]) * (pbb[3] - pbb[1])
                        iou = inter / (areaA + areaB - inter + 1e-6)
                        if iou > max_iou: max_iou = iou

                    rad = 3.0 + (oa['depth']+ob['depth'])*0.05
                    if oa['is_turning'] or ob['is_turning']: rad += 1.5
                    elif oa['is_lateral'] or ob['is_lateral']: rad += 0.8
                    
                    prob = 0.0
                    if oa['is_stationary'] and ob['is_stationary']: prob = 0.0
                    else:
                        if min_dist < rad:
                            prob += 40.0 + (1.0/(time_at_min+0.5))*10 + (1.0/(min_dist+0.5))*10 + impact_v * 1.5
                        elif min_dist < rad + 3.0:
                            prob += 15.0 + (1.0/(min_dist-rad+0.5))*5
                        
                        if max_iou > 0.01: prob += min(30.0, max_iou*150)
                        
                        cxA, cyA = (oa['box'][0]+oa['box'][2])/2, (oa['box'][1]+oa['box'][3])/2
                        cxB, cyB = (ob['box'][0]+ob['box'][2])/2, (ob['box'][1]+ob['box'][3])/2
                        rel_vx, rel_vy = ob['box_vel'][0]-oa['box_vel'][0], ob['box_vel'][1]-oa['box_vel'][1]
                        dist_pix = np.sqrt((cxA-cxB)**2 + (cyA-cyB)**2) + 1e-6
                        if ((cxB-cxA)*rel_vx + (cyB-cyA)*rel_vy)/dist_pix < -5: prob += 10.0
                        
                        if oa['is_turning'] or ob['is_turning']: prob += 15.0
                        elif (oa['is_lateral'] or ob['is_lateral']) and (not oa['is_stationary'] or not ob['is_stationary']): prob += 8.0
                        
                        if oa['is_stationary'] or ob['is_stationary']: prob *= 0.7

                    if prob > 25.0: current_collisions.append(min(99.9, prob))

            max_prob = max(current_collisions) if current_collisions else 0.0
            last_frame_package = {"prob": max_prob}

        return last_frame_package

def main():
    # Setup Logging
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, datetime.now().strftime("%m%d%H%M.log"))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    logger = logging.getLogger()
    
    pipeline = Pipeline()
    if not os.path.exists(ANNOT_FILE):
        logger.error(f"Annotation file {ANNOT_FILE} not found.")
        return
    df = pd.read_csv(ANNOT_FILE)
    
    y_true, y_pred, y_score = [], [], []
    logger.info(f"Starting Evaluation on {len(df)} videos...")

    pipeline.load_perception_models()
    
    for idx, row in df.iterrows():
        file_name = row['file_name']
        gt_label = int(row['label']) if 'label' in df.columns else int(row['risk'])
        
        if EVAL_TEST:
            frames_dir = os.path.join(TEST_DIR, file_name)
            if not os.path.exists(frames_dir):
                logger.warning(f"Video {file_name} not found in train or test dirs. Skipping.")
                continue
        else:
            frames_dir = os.path.join(TRAIN_DIR, file_name)
            if not os.path.exists(frames_dir):
                logger.warning(f"Video {file_name} not found in train or test dirs. Skipping.")
                continue
            
            
        result = pipeline.analyze_frames(frames_dir)
        if result:
            prob = result['prob']
            pred = 1 if prob >= 50.0 else 0
            y_true.append(gt_label)
            y_pred.append(pred)
            y_score.append(prob / 100.0)
            logger.info(f"Video: {file_name} | GT: {gt_label} | Pred Prob: {prob:.1f}% | Pred: {pred}")
        
    if y_true:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_score)
        except:
            auc = 0.0
            
        logger.info("\n" + "="*30)
        logger.info(f"FINAL METRICS")
        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"AUC Score: {auc:.4f}")
        logger.info("="*30)

if __name__ == "__main__":
    main()
