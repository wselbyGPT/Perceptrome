// client/src/main.ts
import { getMe, logout, type Me } from "./auth_api";
import { setupPerceptromeViz } from "./perceptrome_viz";

function buildNextUrl(): string {
  return encodeURIComponent(
    window.location.pathname + window.location.search + window.location.hash
  );
}

function redirectToLogin(): void {
  window.location.href = `/login.html?next=${buildNextUrl()}`;
}

function setUserUi(me: Me): void {
  const whoamiEl = document.getElementById("whoami");
  if (whoamiEl) {
    whoamiEl.textContent = `${me.email} (${me.role})`;
  }

  const logoutBtn = document.getElementById("logout-btn");
  if (logoutBtn) {
    const cloned = logoutBtn.cloneNode(true) as HTMLElement;
    logoutBtn.parentNode?.replaceChild(cloned, logoutBtn);

    cloned.addEventListener("click", async () => {
      try {
        await logout();
      } catch (err) {
        console.warn("Logout failed:", err);
      } finally {
        window.location.href = "/login.html";
      }
    });
  }
}

function makeWebSocketUrl(): string {
  const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${wsProto}//${window.location.host}/ws`;
}

function connectAuthenticatedWebSocket(): WebSocket {
  const ws = new WebSocket(makeWebSocketUrl());

  ws.addEventListener("open", () => {
    console.log("[ws] connected");
  });

  ws.addEventListener("error", (ev) => {
    console.warn("[ws] error", ev);
  });

  ws.addEventListener("close", (ev) => {
    console.warn(`[ws] closed code=${ev.code} reason=${ev.reason || "(none)"}`);
    if (ev.code === 4401) {
      redirectToLogin();
      return;
    }
  });

  return ws;
}

async function requireAuthenticatedUser(): Promise<Me> {
  try {
    return await getMe();
  } catch (err) {
    console.warn("[auth] not authenticated or auth check failed:", err);
    redirectToLogin();
    throw err;
  }
}

async function boot(): Promise<void> {
  if (window.location.pathname.endsWith("/login.html")) {
    return;
  }

  const me = await requireAuthenticatedUser();
  setUserUi(me);

  const ws = connectAuthenticatedWebSocket();
  setupPerceptromeViz(ws);
}

function start(): void {
  void boot().catch((err) => {
    console.error("[boot] failed:", err);
  });
}

if (document.readyState === "loading") {
  window.addEventListener("DOMContentLoaded", start);
} else {
  start();
}
