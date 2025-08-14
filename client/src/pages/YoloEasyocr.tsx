import { useOutletContext } from "react-router";
import FormExtract from "../components/extract/FormExtract";

export function Component() {
	const [pageTitle] = useOutletContext() as any;
	return (
		<FormExtract
			pageTitle={pageTitle}
			requestUrl="/api/cccd/extract/yolo-easyocr"
		/>
	);
}
