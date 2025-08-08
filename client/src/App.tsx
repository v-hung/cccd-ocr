import { useEffect } from "react";
import { Outlet, useLocation } from "react-router";
import "./App.css";

async function loadPreline() {
  return import("preline/dist/index.js");
}

function App() {
  const location = useLocation();

  useEffect(() => {
    const initPreline = async () => {
      await loadPreline();

      if (
        window.HSStaticMethods &&
        typeof window.HSStaticMethods.autoInit === "function"
      ) {
        window.HSStaticMethods.autoInit();
      }
    };

    initPreline();
  }, [location.pathname]);

  return <Outlet />;
}

export default App;
