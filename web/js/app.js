import { boot } from "./boot.js";
import { Toasts } from "./toasts.js";

document.addEventListener("DOMContentLoaded", () => {
  boot().catch((e) => Toasts.toast(`Boot failed: ${e.message}`, "error", 6000));
});

window.addEventListener("unhandledrejection", (e) => {
  try {
    Toasts.show({ type: "error", title: "Unhandled error", message: String(e.reason || e), ms: 4800 });
  } catch {}
});
