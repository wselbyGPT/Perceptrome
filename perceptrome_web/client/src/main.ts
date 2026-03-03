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

function redirectToChangePassword(): void {
  window.location.href = `/change_password.html?next=${buildNextUrl()}`;
}

function setUserUi(me: Me): void {
  const whoamiEl = document.getElementById("whoami");
  if (whoamiEl) {
    whoamiEl.textContent = `${me.email} (${me.role})`;
  }

  const changePasswordBtn = document.getElementById("change-password-btn");
  if (changePasswordBtn) {
    const changePasswordClone = changePasswordBtn.cloneNode(true) as HTMLElement;
    changePasswordBtn.parentNode?.replaceChild(changePasswordClone, changePasswordBtn);
    changePasswordClone.addEventListener("click", () => {
      window.location.href = `/change_password.html?next=${buildNextUrl()}`;
    });
  }

  const logoutBtn = document.getElementById("logout-btn");
  if (logoutBtn) {
    const logoutClone = logoutBtn.cloneNode(true) as HTMLElement;
    logoutBtn.parentNode?.replaceChild(logoutClone, logoutBtn);

    logoutClone.addEventListener("click", async () => {
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

    if (ev.code === 4403) {
      redirectToChangePassword();
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
  if (me.must_change_password) {
    redirectToChangePassword();
    return;
  }

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
