import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // During `npm run dev`, forward /api calls to a locally running
      // `vercel dev` (or `uvicorn api.index:app --port 8000`) instance.
      "/api": "http://localhost:3000",
    },
  },
});
