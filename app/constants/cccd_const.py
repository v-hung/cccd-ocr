REGION_CONFIG = {
	"id": {
		"box": [341, 211, 298, 60],
		"description": "Mã số định danh cá nhân (12 chữ số)."
	},
	"name": {
		"box": [244, 289, 336, 43],
		"description": "Họ tên đầy đủ in hoa không dấu."
	},
	"dob": {
		"box": [492, 328, 151, 38],
		"description": "Ngày sinh theo định dạng dd/mm/yyyy."
	},
	"gender": {
		"box": [397, 360, 77, 39],
		"description": "Thông tin giới tính: Nam hoặc Nữ."
	},
	"origin_place": {
		"box": [238, 420, 432, 46],
		"description": "Địa phương gốc: xã/phường, huyện/quận, tỉnh/thành."
	}
}

REGION_FIELDS = set(REGION_CONFIG.keys())

REGION_FIELDS_BOX = {
	key: value["box"]
	for key, value in REGION_CONFIG.items()
}

REGION_FIELDS_DESCRIPTION = {
	key: value["description"]
	for key, value in REGION_CONFIG.items()
}