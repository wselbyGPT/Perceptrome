import "./styles/index.css";
// client/src/admin_users.ts
import {
  adminCreateUser,
  adminListUsers,
  getMe,
  logout,
  type AdminUser,
} from "./auth_api";

function redirectToLogin() {
  const next = encodeURIComponent(window.location.pathname + window.location.search + window.location.hash);
  window.location.href = `/login.html?next=${next}`;
}

function redirectToChangePassword() {
  const next = encodeURIComponent(window.location.pathname + window.location.search + window.location.hash);
  window.location.href = `/change_password.html?next=${next}`;
}

function randomTempPassword(length = 18): string {
  const chars = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789!@#$%^&*";
  let out = "";
  const arr = new Uint32Array(length);
  crypto.getRandomValues(arr);
  for (let i = 0; i < length; i++) {
    out += chars[arr[i] % chars.length];
  }
  return out;
}

function pill(text: string, className: string): string {
  return `<span class="badge badge-${className}">${text}</span>`;
}

function renderUsers(rows: AdminUser[]) {
  const tbody = document.getElementById("users-tbody") as HTMLTableSectionElement;
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="6" class="muted">No users found.</td></tr>`;
    return;
  }

  tbody.innerHTML = rows
    .map((u) => {
      const roleClass = u.role === "admin" ? "admin" : "user";
      const activeClass = u.is_active ? "active" : "inactive";
      const forceClass = u.must_change_password ? "force" : "noforce";
      const username = u.username ?? "";
      return `
        <tr>
          <td>${u.email}</td>
          <td>${username}</td>
          <td>${pill(u.role, roleClass)}</td>
          <td>${pill(u.is_active ? "active" : "inactive", activeClass)}</td>
          <td>${pill(u.must_change_password ? "must change" : "normal", forceClass)}</td>
          <td class="mono">${u.id}</td>
        </tr>
      `;
    })
    .join("");
}

function setMsg(id: string, message: string, type: "ok" | "error" | "plain" = "plain") {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = message;
  el.className = "msg" + (type === "plain" ? "" : ` ${type}`);
}

async function loadUsers() {
  setMsg("table-msg", "Loading users…");
  try {
    const users = await adminListUsers();
    renderUsers(users);
    setMsg("table-msg", `Loaded ${users.length} user(s).`, "ok");
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Failed to load users";
    setMsg("table-msg", msg, "error");
    throw err;
  }
}

async function boot() {
  const me = await getMe().catch(() => {
    redirectToLogin();
    throw new Error("Not authenticated");
  });

  if (me.must_change_password) {
    redirectToChangePassword();
    return;
  }

  if (me.role !== "admin") {
    alert("Admin access required.");
    window.location.href = "/runs.html";
    return;
  }

  const whoamiEl = document.getElementById("whoami");
  if (whoamiEl) whoamiEl.textContent = `${me.email} (${me.role})`;

  const logoutBtn = document.getElementById("logout-btn") as HTMLButtonElement | null;
  logoutBtn?.addEventListener("click", async () => {
    try { await logout(); } catch {}
    window.location.href = "/login.html";
  });

  const emailEl = document.getElementById("email") as HTMLInputElement;
  const usernameEl = document.getElementById("username") as HTMLInputElement;
  const passwordEl = document.getElementById("password") as HTMLInputElement;
  const roleEl = document.getElementById("role") as HTMLSelectElement;
  const isActiveEl = document.getElementById("is-active") as HTMLInputElement;
  const mustChangeEl = document.getElementById("must-change-password") as HTMLInputElement;
  const createBtn = document.getElementById("create-btn") as HTMLButtonElement;
  const genPassBtn = document.getElementById("gen-pass-btn") as HTMLButtonElement;
  const refreshBtn = document.getElementById("refresh-btn") as HTMLButtonElement;
  const form = document.getElementById("create-user-form") as HTMLFormElement;

  if (!passwordEl.value) passwordEl.value = randomTempPassword();

  genPassBtn.addEventListener("click", () => {
    passwordEl.value = randomTempPassword();
  });

  refreshBtn.addEventListener("click", () => {
    void loadUsers();
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    setMsg("form-msg", "");
    createBtn.disabled = true;

    try {
      const created = await adminCreateUser({
        email: emailEl.value.trim(),
        username: usernameEl.value.trim() || null,
        password: passwordEl.value,
        role: (roleEl.value as "user" | "admin") ?? "user",
        is_active: isActiveEl.checked,
        must_change_password: mustChangeEl.checked,
      });

      setMsg(
        "form-msg",
        `Created user: ${created.email}${created.must_change_password ? " (must change password on first login)" : ""}`,
        "ok"
      );

      emailEl.value = "";
      usernameEl.value = "";
      passwordEl.value = randomTempPassword();
      roleEl.value = "user";
      isActiveEl.checked = true;
      mustChangeEl.checked = true;

      await loadUsers();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to create user";
      setMsg("form-msg", msg, "error");
    } finally {
      createBtn.disabled = false;
    }
  });

  await loadUsers();
}

if (document.readyState === "loading") {
  window.addEventListener("DOMContentLoaded", () => void boot());
} else {
  void boot();
}
