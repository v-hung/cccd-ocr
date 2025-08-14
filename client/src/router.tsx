import { createBrowserRouter, Navigate } from "react-router";
import App from "./App";

const router = createBrowserRouter([
	{
		element: <App />,
		children: [
			{
				lazy: () => import("./layouts/DefaultLayout"),
				children: [
					{
						path: "/",
						element: <Navigate to="/opencv-easyocr" replace />,
					},
					{
						path: "/opencv-easyocr",
						lazy: () => import("./pages/OpencvEasyocr"),
					},
					{
						path: "/yolo-easyoc",
						lazy: () => import("./pages/YoloEasyocr"),
					},
					{
						path: "/gemini",
						lazy: () => import("./pages/Gemini"),
					},
				],
			},
		],
	},
]);

export default router;
