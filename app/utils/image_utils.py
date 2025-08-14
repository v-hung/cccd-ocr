import cv2
import numpy as np
from pathlib import Path

def load_image(image_input):
	if isinstance(image_input, (str, Path)):
		image = cv2.imread(str(image_input))
		if image is None:
			raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image_input}")
		return image

	elif isinstance(image_input, (bytes, bytearray)):
		image_array = np.frombuffer(image_input, np.uint8)
		image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
		if image is None:
			raise ValueError("Không thể giải mã ảnh từ bytes.")
		return image

	else:
		raise TypeError("Đầu vào không hợp lệ. Phải là đường dẫn (str/Path) hoặc bytes.")
