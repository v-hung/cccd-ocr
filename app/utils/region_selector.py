import cv2
import numpy as np
from pathlib import Path

class RegionSelector:
	def __init__(self, image_input):
		if isinstance(image_input, (str, Path)):
			image = cv2.imread(str(image_input))
			if image is None:
				raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image_input}")
		elif isinstance(image_input, np.ndarray):
			image = image_input
		else:
			raise TypeError("Tham số 'image_input' phải là đường dẫn (str) hoặc ảnh (numpy.ndarray)")

		self.image = image.copy()
		self.clone = image.copy()
		self.regions = []
		self.drawing = False
		self.start_point = (0, 0)
		self.current_point = (0, 0)

	def select_regions(self):
		cv2.namedWindow("Select Regions")
		cv2.setMouseCallback("Select Regions", self.mouse_callback)

		while True:
			if cv2.getWindowProperty("Select Regions", cv2.WND_PROP_VISIBLE) < 1:
				break
	
			display = self.image.copy()

			for (x, y, w, h) in self.regions:
				cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

			if self.drawing:
				x1, y1 = self.start_point
				x2, y2 = self.current_point
				cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)

			cv2.imshow("Select Regions", display)
			key = cv2.waitKey(1) & 0xFF

			if key == ord("q"):
				break
			elif key == ord("r"):
				self.regions.clear()
				self.image = self.clone.copy()

		for i, (x, y, w, h) in enumerate(self.regions):
			print(f"{i + 1}: x={x}, y={y}, w={w}, h={h}")

		cv2.destroyAllWindows()
		return self.regions

	def mouse_callback(self, event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			self.drawing = True
			self.start_point = (x, y)
			self.current_point = (x, y)

		elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
			self.current_point = (x, y)

		elif event == cv2.EVENT_LBUTTONUP:
			self.drawing = False
			x1, y1 = self.start_point
			x2, y2 = self.current_point
			x, y = min(x1, x2), min(y1, y2)
			w, h = abs(x2 - x1), abs(y2 - y1)
			if w > 5 and h > 5:
				self.regions.append((x, y, w, h))
				print(f"Thêm vùng: x={x}, y={y}, w={w}, h={h}")

		elif event == cv2.EVENT_RBUTTONDOWN:
			self.remove_region_at(x, y)

	def remove_region_at(self, x, y):
		for i in range(len(self.regions) - 1, -1, -1):
			rx, ry, rw, rh = self.regions[i]
			if rx <= x <= rx + rw and ry <= y <= ry + rh:
				print(f"Xoá vùng: x={rx}, y={ry}, w={rw}, h={rh}")
				self.regions.pop(i)
				break