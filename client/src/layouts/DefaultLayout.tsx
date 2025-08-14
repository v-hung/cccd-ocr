import { useEffect, useMemo, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router";
import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

const useMenuStore = create(
	persist<{ isMenuOpen: boolean }>(
		(set) => ({
			isMenuOpen: true,
		}),
		{
			name: "menu-storage",
			storage: createJSONStorage(() => sessionStorage),
		},
	),
);

export function Component() {
	const isMenuOpen = useMenuStore((state) => state.isMenuOpen);

	const menu = [
		{
			path: "/opencv-easyocr",
			icon: <IIonCameraOutline className="shrink-0" />,
			label: "OpenCV + EasyOCR",
		},
		{
			path: "/yolo-easyoc",
			icon: <IIonScan className="shrink-0" />,
			label: "YOLO + EasyOCR",
		},
		{
			path: "/gemini",
			icon: <IIonLogoGoogle className="shrink-0" />,
			label: "Gemini",
		},
	];

	const location = useLocation();

	const pageTitle = useMemo(
		() => menu.find((v) => v.path == location.pathname)?.label || "",
		[location],
	);

	return (
		<div className="flex min-h-full w-full flex-1 flex-col bg-gray-50">
			<header className="sticky top-0 z-50 flex h-16 flex-none items-center gap-4 bg-gradient-to-r from-sky-700 via-sky-700 to-sky-600 px-2 py-2 font-semibold text-white shadow-sm">
				<button
					type="button"
					className="flex size-11 items-center justify-center"
					onClick={() =>
						useMenuStore.setState((state) => ({
							isMenuOpen: !state.isMenuOpen,
						}))
					}
				>
					{isMenuOpen ? (
						<IIonMenu className="shrink-0" />
					) : (
						<IIonClose className="shrink-0" />
					)}
				</button>

				<h1 className="text-lg capitalize">{pageTitle}</h1>
			</header>

			<div className="flex min-h-0 flex-grow flex-row">
				<div
					className={`group fixed top-16 bottom-0 z-50 flex flex-none flex-col gap-y-1 border-r border-gray-200 bg-white px-2 py-2 shadow transition-[width] ${
						isMenuOpen ? "is-closed w-16" : "w-80"
					}`}
				>
					{menu.map((v) => (
						<NavLink
							key={v.path}
							to={v.path}
							className={({ isActive }) =>
								`flex flex-nowrap items-center overflow-hidden rounded hover:bg-blue-50 ${isActive ? "bg-sky-100!" : ""}`
							}
						>
							<div className="flex size-11 flex-none items-center justify-center">
								{v.icon}
							</div>
							<span className="min-w-0 flex-grow whitespace-nowrap group-[.is-closed]:hidden">
								{v.label}
							</span>
						</NavLink>
					))}
				</div>
				<div
					className={`flex min-h-0 flex-grow transition-all ${isMenuOpen ? "ml-16" : "ml-80"}`}
				>
					<div className="w-full px-8 py-4">
						<Outlet context={[pageTitle]} />
					</div>
				</div>
			</div>
		</div>
	);
}
