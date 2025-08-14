import json, re

def parse_json_block(text: str) -> dict:
	clean = re.sub(r"^```json\s*|\s*```$", "", text.strip())
	return json.loads(clean)