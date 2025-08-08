from ultralytics import YOLO
import datetime
import os
import cv2
from configs import BASE_DIR, UPLOAD_DIR
from constants.cccd_const import REGION_FIELDS

MODEL_PATH = BASE_DIR / "ai_models/cccd_yolo/weights/best.pt"

model = YOLO(MODEL_PATH)

def extract_yolo_regions(results) -> dict:
    now = datetime.datetime.now()
    save_folder = UPLOAD_DIR / 'temp' / now.strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_folder, exist_ok=True)

    image = results[0].orig_img
    image_regions = image.copy()
    detections = results[0].boxes
    names = results[0].names

    cropped_map = {}

    for box in detections:
        cls_id = int(box.cls[0].item())
        label = names[cls_id]
        
        if label in REGION_FIELDS:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            cropped = image[y1:y2, x1:x2]
            cropped_map[label] = cropped

            cv2.rectangle(image_regions, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(str(save_folder / '1_original_input.jpg'), image)
    cv2.imwrite(str(save_folder / '2_regions_input.jpg'), image_regions)

    return cropped_map