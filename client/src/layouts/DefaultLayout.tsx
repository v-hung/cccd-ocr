import { useState } from "react";
import { Outlet } from "react-router";
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

  return (
    <div className="flex h-full w-full flex-col bg-gray-50">
      <header className="sticky top-0 flex flex-none items-center gap-4 bg-gradient-to-r from-sky-700 via-sky-700 to-sky-600 px-2 py-2 font-semibold text-white shadow-sm">
        <button
          type="button"
          className="flex size-11 items-center justify-center"
          onClick={() =>
            useMenuStore.setState((state) => ({
              isMenuOpen: !state.isMenuOpen,
            }))
          }
        >
          <IIonMenu className="shrink-0" />
        </button>

        <h1 className="text-lg capitalize">f asdf asdf sa dfa dsf a</h1>
      </header>

      <div className="flex min-h-0 flex-grow flex-row">
        <div
          className={`group flex flex-none flex-col gap-y-1 border-r border-gray-200 bg-white px-2 py-2 shadow transition-[width] ${
            isMenuOpen ? "is-closed w-16" : "w-80"
          }`}
        >
          <a
            href="#"
            className="flex flex-nowrap items-center overflow-hidden rounded bg-sky-100! hover:bg-blue-50"
            onClick={() =>
              useMenuStore.setState((state) => ({
                isMenuOpen: !state.isMenuOpen,
              }))
            }
          >
            <div className="flex size-11 flex-none items-center justify-center">
              <IIonMenu className="shrink-0" />
            </div>
            <span className="min-w-0 flex-grow whitespace-nowrap group-[.is-closed]:hidden">
              fasdfsadf f adsf asdf
            </span>
          </a>
          <a
            href="#"
            className="flex flex-nowrap items-center overflow-hidden rounded hover:bg-blue-50"
            onClick={() =>
              useMenuStore.setState((state) => ({
                isMenuOpen: !state.isMenuOpen,
              }))
            }
          >
            <div className="flex size-11 flex-none items-center justify-center">
              <IIonMenu className="shrink-0" />
            </div>
            <span className="min-w-0 flex-grow whitespace-nowrap group-[.is-closed]:hidden">
              fasdfsadf f adsf asdf
            </span>
          </a>
        </div>
        <div className="flex min-h-0 flex-grow">
          <div className="w-full p-4">
            <Outlet />
          </div>
        </div>
      </div>
    </div>
  );
}
