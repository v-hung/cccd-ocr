import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import tailwindcss from "@tailwindcss/vite";

import Icons from "unplugin-icons/vite";
import AutoImport from "unplugin-auto-import/vite";
import IconsResolver from "unplugin-icons/resolver";

// https://vite.dev/config/
export default defineConfig({
	plugins: [
		react(),
		tailwindcss(),
		AutoImport({
			dts: "src/types/components.d.ts",
			resolvers: [
				IconsResolver({
					// prefix: false,
					extension: "jsx",
				}),
			],
		}),
		Icons({
			autoInstall: true,
			compiler: "jsx",
			jsx: "react",
			iconCustomizer(_, __, props) {
				props.width = "24px";
				props.height = "24px";
			},
		}),
	],
	// build: {
	// 	outDir: "../app/static",
	// 	emptyOutDir: true,
	// },
	server: {
		proxy: {
			"^/api": { target: "http://localhost:8000", secure: false },
			"^/uploads": { target: "http://localhost:8000", secure: false },
		},
	},
});
