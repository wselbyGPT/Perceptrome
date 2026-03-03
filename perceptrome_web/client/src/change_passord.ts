// client/src/change_password.ts
import { changePassword, getMe, logout } from "./auth_api";

function redirectToLogin() {
  const next = encodeURIComponent(window.location.pathname + window.location.search + window.location.hash);
  window.location.href = `/login.html?next=${next}`;
}

function getNextDestination(): string {
  const raw = new URLSearchParams(window.location.search).get("next");
  return raw || "/";
}

function setMsg(message: string, type: "plain" | "error" | "ok" = "plain") {
  const el = document.getElementById("msg");
  if (!el) return;
  el.textContent = message;
  el.className = "msg" + (type === "plain" ? "" : ` ${type}`);
}

function setPasswordFieldsVisible(visible: boolean) {
  const ids = ["current-password", "new-password", "confirm-password"] as const;
  for (const id of ids) {
    const el = document.getElementById(id) as HTMLInputElement | null;
    if (el) el.type = visible ? "text" : "password";
  }
  const toggleBtn = document.getElementById("toggle-btn") as HTMLButtonElement | null;
  if (toggleBtn) toggleBtn.textContent = visible ? "Hide" : "Show";
}

async function boot() {
  const me = await getMe().catch(() => {
    redirectToLogin();
    throw new Error("Not authenticated");
  });

  const whoamiEl = document.getElementById("whoami");
  if (whoamiEl) {
    whoamiEl.textContent = `${me.email} (${me.role})`;
  }

  const form = document.getElementById("change-password-form") as HTMLFormElement;
  const currentEl = document.getElementById("current-password") as HTMLInputElement;
  const newEl = document.getElementById("new-password") as HTMLInputElement;
  const confirmEl = document.getElementById("confirm-password") as HTMLInputElement;
  const submitBtn = document.getElementById("submit-btn") as HTMLButtonElement;
  const logoutBtn = document.getElementById("logout-btn") as HTMLButtonElement;
  const toggleBtn = document.getElementById("toggle-btn") as HTMLButtonElement;

  let visible = false;
  toggleBtn.addEventListener("click", () => {
    visible = !visible;
    setPasswordFieldsVisible(visible);
  });

  logoutBtn.addEventListener("click", async () => {
    try {
      await logout();
    } catch {
      // ignore
    } finally {
      window.location.href = "/login.html";
    }
  });

  if (!me.must_change_password) {
    setMsg("Password already updated. Redirecting…", "ok");
    window.setTimeout(() => {
      window.location.href = getNextDestination();
    }, 500);
    return;
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    setMsg("");
    submitBtn.disabled = true;

    try {
      const currentPassword = currentEl.value;
      const newPassword = newEl.value;
      const confirmPassword = confirmEl.value;

      if (newPassword.length < 8) {
        throw new Error("New password must be at least 8 characters");
      }
      if (newPassword !== confirmPassword) {
        throw new Error("New passwords do not match");
      }

      await changePassword(currentPassword, newPassword);
      setMsg("Password updated. Redirecting…", "ok");

      window.setTimeout(() => {
        window.location.href = getNextDestination();
      }, 700);
    } catch (err) {
      setMsg(err instanceof Error ? err.message : "Failed to update password", "error");
    } finally {
      submitBtn.disabled = false;
    }
  });
}

if (document.readyState === "loading") {
  window.addEventListener("DOMContentLoaded", () => {
    void boot();
  });
} else {
  void boot();
}
