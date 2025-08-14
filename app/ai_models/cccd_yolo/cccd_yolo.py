from ultralytics import YOLO
import datetime
import os
import cv2
from configs import BASE_DIR, UPLOAD_DIR
from constants.cccd_const import REGION_FIELDS
from pathlib import Path

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

			cv2.rectangle(image_regions, (x1, y1), (x2, y2), (0, 255, 0), 5)

	cv2.imwrite(str(save_folder / '1_original_input.jpg'), image)
	cv2.imwrite(str(save_folder / '2_regions_input.jpg'), image_regions)

	content = generate_content(save_folder)

	return (cropped_map, content)

def generate_content(folder_path: Path):
	parts = folder_path.parts
	idx = parts.index('uploads')
	relative_path = Path(*parts[idx:])
	base_path = str(relative_path)
	
	return f'''
	<div>
		<h3 style="margin-top:0">Quy trình xử lý ảnh CCCD và trích xuất thông tin</h3>
        
		<h4>Bước 1: Xử lý ảnh</h4>
		<div style="margin-bottom: 20px;">
			<p style="margin-bottom: .5rem"><strong>Ảnh 1 - Ảnh gốc:</strong> Ảnh đầu vào nguyên bản chưa qua xử lý.</p>
			<img src="{base_path}/1_original_input.jpg" alt="Ảnh gốc" style="max-width: 500px; border: 1px solid #ccc;margin-top:0" />
		</div>

		<div style="margin-bottom: 20px;">
			<p style="margin-bottom: .5rem"><strong>Ảnh 2 - Vùng ảnh nhập OCR:</strong> Nhận diện các vùng lấy thông tin thông qua yolo model (Object detection).</p>
			<img src="{base_path}/2_regions_input.jpg" alt="Vùng ảnh cho OCR" style="max-width: 500px; border: 1px solid #ccc;margin-top:0" />
		</div>

		<h4>Bước 2: Đọc thông tin qua OCR</h4>
		<p>Sau khi có được các vùng ảnh đã chuẩn bị ở bước 1, EasyOCR sẽ được sử dụng để trích xuất văn bản từ từng vùng.</p>
		<p>Dữ liệu thu thập từ OCR sẽ được xử lý, xác thực và tổng hợp thành dữ liệu đầu ra cuối cùng dưới dạng JSON hoặc các định dạng phù hợp.</p>
	</div>
	'''