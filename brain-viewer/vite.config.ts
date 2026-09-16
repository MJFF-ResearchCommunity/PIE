import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { apiPort } from "./src/apiTarget";

// Match `python -m pie.imaging.viewer serve --port N` with PIE_VIEWER_API_PORT=N.
const port = apiPort(process.env.PIE_VIEWER_API_PORT);

export default defineConfig({
  plugins: [react()],
  define: { __PIE_API_PORT__: JSON.stringify(port) },
  server: {
    port: 5178,
    strictPort: true,
    proxy: { "/api": `http://127.0.0.1:${port}` },
  },
  build: { chunkSizeWarningLimit: 2500 },
});
