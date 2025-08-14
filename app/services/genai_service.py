from google import genai
from google.genai import types
from configs import GEMINI_KEY
from utils.image_utils import load_image
from constants.cccd_const import REGION_FIELDS_DESCRIPTION
import cv2
import json
from utils.json_utils import parse_json_block

client = genai.Client(api_key=GEMINI_KEY)

def extract_cccd_info_with_openai(image_input):

	image = load_image(image_input)
	image_bytes = cv2.imencode(".jpg", image)[1].tobytes()

	image_data = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

	prompt = (
		f"Trích xuất các trường thông tin trong ảnh sau và trả kết quả dưới dạng JSON.\n"
		"Vui lòng trả kết quả theo đúng định dạng JSON như ví dụ:\n"
		f"{json.dumps(REGION_FIELDS_DESCRIPTION)}"
	)

	response = client.models.generate_content(
		model='gemini-2.0-flash',
		contents=[prompt, image_data]
	)

	return parse_json_block(response.text)