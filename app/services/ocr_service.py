import easyocr

def ocr_regions_easyocr(cropped_map: dict, lang_list=['vi']) -> dict:
	reader = easyocr.Reader(lang_list, gpu=False)
	ocr_result = {}

	for key, image in cropped_map.items():
		results = reader.readtext(image, detail=0, paragraph=True)
		text = " ".join(results).strip() if results else ""
		ocr_result[key] = text

	return ocr_result