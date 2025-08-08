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
    cropped_map = process_cccd_image(content)

    results = ocr_regions_easyocr(cropped_map)
    return results

@router.post("/extract/yolo-easyocr")
async def extract_cccd_opencv_easyocr(file: UploadFile = File(...)):
    content = await file.read()
    image = load_image(content)
    model_results = model(image)

    cropped_map = extract_yolo_regions(model_results)

    results = ocr_regions_easyocr(cropped_map)
    return results

@router.post("/extract/gemini")
async def extract_cccd_opencv_easyocr(file: UploadFile = File(...)):
    content = await file.read()
    results = extract_cccd_info_with_openai(content)
    
    return results
