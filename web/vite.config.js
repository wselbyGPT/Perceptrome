import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

async function httpOk(url, timeoutMs = 500) {
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), timeoutMs);
    const res = await fetch(url, { signal: ctrl.signal });
    clearTimeout(t);
    return res.ok;
  } catch {
    return false;
  }
}

async function detectApiMode(targetBase) {
  // Prefer "keep" if /api/openapi.json works; otherwise "strip" if /openapi.json works.
  if (await httpOk(`${targetBase}/api/openapi.json`)) return "keep";
  if (await httpOk(`${targetBase}/openapi.json`)) return "strip";
  // Default: strip (matches your nginx auto-detection style)
  return "strip";
}

export default defineConfig(async () => {
  const target = process.env.VITE_DEV_API_TARGET || "http://127.0.0.1:9000";
  const mode = await detectApiMode(target);

  return {
    plugins: [react()],
    server: {
      proxy: {
        "/api": {
          target,
          changeOrigin: true,
          rewrite:
            mode === "strip"
              ? (path) => path.replace(/^\/api/, "")
              : (path) => path
        }
      }
    }
  };
});
