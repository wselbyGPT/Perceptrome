// client/vite.config.ts
import { defineConfig } from "vite";
import { resolve } from "node:path";

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        index: resolve(__dirname, "index.html"),
        login: resolve(__dirname, "login.html"),
        admin_users: resolve(__dirname, "admin_users.html"),
        datasets: resolve(__dirname, "datasets.html"),
        change_password: resolve(__dirname, "change_password.html"),
        verify_email: resolve(__dirname, "verify_email.html"),
        forgot_password: resolve(__dirname, "forgot_password.html"),
        reset_password: resolve(__dirname, "reset_password.html"),
      },
    },
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        secure: false,
      },
      "/ws": {
        target: "http://127.0.0.1:8000",
        ws: true,
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
