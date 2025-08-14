from fastapi import APIRouter, File, UploadFile
from utils.cccd_image_utils import process_cccd_image
from services.ocr_service import ocr_regions_easyocr
from ai_models.cccd_yolo.cccd_yolo import model, extract_yolo_regions
from services.genai_service import extract_cccd_info_with_openai
from utils.image_utils import load_image

router = APIRouter()

@router.post("/extract/opencv-easyocr")
async def extract_cccd_opencv_easyocr(file: UploadFile = File(...)):
	content = await file.read()
	cropped_map, content = process_cccd_image(content)

	result = ocr_regions_easyocr(cropped_map)
	return { "extract": result, "content": content}

@router.post("/extract/yolo-easyocr")
async def extract_cccd_yolo_easyocr(file: UploadFile = File(...)):
	content = await file.read()
	image = load_image(content)
	model_result = model(image)

	cropped_map, content = extract_yolo_regions(model_result)

	result = ocr_regions_easyocr(cropped_map)
	return { "extract": result, "content": content}

@router.post("/extract/gemini")
async def extract_cccd_gemini(file: UploadFile = File(...)):
	content = await file.read()
	result = extract_cccd_info_with_openai(content)
	
	return { "extract": result}
