import { createBrowserRouter } from "react-router";
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
            lazy: () => import("./pages/HomePage"),
          },
        ],
      },
    ],
  },
]);

export default router;
