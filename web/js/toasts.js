import { U } from "./utils.js";

export const Toasts = (() => {
  let host = null;

  function ensureHost() {
    host = U.$("#toastHost");
    if (!host) {
      host = document.createElement("div");
      host.id = "toastHost";
      host.className = "toasts";
      host.setAttribute("aria-live", "polite");
      host.setAttribute("aria-atomic", "true");
      document.body.appendChild(host);
    }
    return host;
  }

  function iconFor(type) {
    if (type === "success") return "✓";
    if (type === "error") return "!";
    if (type === "warn") return "⚠";
    return "•";
  }

  function show({ type = "info", title = "Info", message = "", ms = 3200 } = {}) {
    ensureHost();

    const node = document.createElement("div");
    node.className = "toast";
    node.dataset.type = type;

    const bar = document.createElement("div");
    bar.className = "toast-bar";
    const barFill = document.createElement("i");
    bar.appendChild(barFill);

    node.innerHTML = `
      <div class="toast-inner">
        <div class="toast-ico" aria-hidden="true">${iconFor(type)}</div>
        <div style="min-width:0">
          <div class="toast-title"></div>
          <div class="toast-msg"></div>
        </div>
        <button class="toast-x" type="button" aria-label="Dismiss">×</button>
      </div>
    `;
    node.querySelector(".toast-title").textContent = title;
    node.querySelector(".toast-msg").textContent = message || "";
    node.appendChild(bar);

    const close = () => {
      node.classList.remove("show");
      setTimeout(() => node.remove(), 180);
    };
    node.querySelector(".toast-x").addEventListener("click", close);

    host.prepend(node);
    requestAnimationFrame(() => node.classList.add("show"));

    if (ms > 0) {
      barFill.style.transition = `transform ${ms}ms linear`;
      requestAnimationFrame(() => (barFill.style.transform = "scaleX(0)"));
      setTimeout(close, ms + 60);
    }
    return node;
  }

  function toast(message, kind = "info", ms = 3200) {
    const map = { good: "success", success: "success", info: "info", warn: "warn", bad: "error", error: "error" };
    const type = map[kind] || "info";
    const title = type === "success" ? "Success" : type === "warn" ? "Warning" : type === "error" ? "Error" : "Info";
    show({ type, title, message: String(message ?? ""), ms: Number(ms || 0) });
  }

  return { show, toast };
})();

// compatibility with older code paths
window.Toasts = window.Toasts || { show: Toasts.show };
