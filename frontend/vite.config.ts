import { defineConfig } from "vite";
import fs from "node:fs";
import path from "node:path";

const backend = "http://127.0.0.1:8000";

export default defineConfig({
  server: {
    host: "::",
    port: 5173,
    https: {
      key: fs.readFileSync(path.resolve(__dirname, "certs/localhost-key.pem")),
      cert: fs.readFileSync(path.resolve(__dirname, "certs/localhost.pem")),
    },
    proxy: {
      "/case-file-intelligence-demo": {
        target: backend,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/case-file-intelligence-demo/, ""),
      },
      "/claims": backend,
      "/health": backend,
      "/docs": backend,
      "/openapi.json": backend,
      "/giec-chat-demo": {
        target: "http://127.0.0.1:8501",
        changeOrigin: true,
        ws: true,
      },
    },
  },
});