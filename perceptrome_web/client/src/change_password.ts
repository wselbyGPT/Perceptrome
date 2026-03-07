import "./styles/index.css";
// client/src/change_password.ts
import { changePassword, getMe, logout } from "./auth_api";

const MIN_PASSWORD_LENGTH = 12;

type ValidationState = {
  currentPassword: string;
  newPassword: string;
  confirmPassword: string;
};

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

function scorePasswordStrength(password: string): { score: number; label: string } {
  if (!password) return { score: 0, label: "Too weak" };

  let score = 0;
  if (password.length >= MIN_PASSWORD_LENGTH) score += 1;
  if (/[A-Z]/.test(password)) score += 1;
  if (/[a-z]/.test(password)) score += 1;
  if (/\d/.test(password)) score += 1;
  if (/[^A-Za-z0-9]/.test(password)) score += 1;

  if (score <= 2) return { score, label: "Weak" };
  if (score <= 4) return { score, label: "Good" };
  return { score, label: "Strong" };
}

function updateFieldError(fieldId: string, errorText: string) {
  const input = document.getElementById(fieldId) as HTMLInputElement | null;
  const err = document.getElementById(`${fieldId}-error`);
  if (!input || !err) return;

  err.textContent = errorText;
  const hasError = Boolean(errorText);
  input.setAttribute("aria-invalid", String(hasError));
}

function validateForm(state: ValidationState): { isValid: boolean; firstErrorFieldId?: string } {
  const errors: Record<string, string> = {
    "current-password": "",
    "new-password": "",
    "confirm-password": "",
  };

  if (!state.currentPassword.trim()) {
    errors["current-password"] = "Enter your current password.";
  }

  if (state.newPassword.length < MIN_PASSWORD_LENGTH) {
    errors["new-password"] = `Use at least ${MIN_PASSWORD_LENGTH} characters.`;
  } else if (state.newPassword === state.currentPassword) {
    errors["new-password"] = "New password must be different from your current password.";
  }

  if (!state.confirmPassword) {
    errors["confirm-password"] = "Please confirm your new password.";
  } else if (state.confirmPassword !== state.newPassword) {
    errors["confirm-password"] = "Passwords do not match.";
  }

  updateFieldError("current-password", errors["current-password"]);
  updateFieldError("new-password", errors["new-password"]);
  updateFieldError("confirm-password", errors["confirm-password"]);

  const firstErrorFieldId = ["current-password", "new-password", "confirm-password"].find((id) => errors[id]);
  return {
    isValid: !firstErrorFieldId,
    firstErrorFieldId,
  };
}

function updateStrengthUi(newPassword: string) {
  const meter = document.getElementById("password-strength") as HTMLProgressElement | null;
  const label = document.getElementById("password-strength-label");
  if (!meter || !label) return;

  const strength = scorePasswordStrength(newPassword);
  meter.value = strength.score;
  label.textContent = `Strength: ${strength.label}`;
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

  const headingEl = document.getElementById("page-title");
  const subtitleEl = document.getElementById("page-subtitle");
  if (headingEl && subtitleEl) {
    if (me.must_change_password) {
      headingEl.textContent = "Password Change Required";
      subtitleEl.textContent = "You must update your password before continuing.";
    } else {
      headingEl.textContent = "Update Your Password";
      subtitleEl.textContent = "Choose a new password to keep your account secure.";
    }
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

  const runInlineValidation = (): boolean => {
    const result = validateForm({
      currentPassword: currentEl.value,
      newPassword: newEl.value,
      confirmPassword: confirmEl.value,
    });
    submitBtn.disabled = !result.isValid;
    return result.isValid;
  };

  newEl.addEventListener("input", () => {
    updateStrengthUi(newEl.value);
    runInlineValidation();
  });

  currentEl.addEventListener("input", runInlineValidation);
  confirmEl.addEventListener("input", runInlineValidation);

  updateStrengthUi("");
  runInlineValidation();

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    setMsg("");

    const { isValid, firstErrorFieldId } = validateForm({
      currentPassword: currentEl.value,
      newPassword: newEl.value,
      confirmPassword: confirmEl.value,
    });

    if (!isValid) {
      setMsg("Please fix the highlighted password issues.", "error");
      if (firstErrorFieldId) {
        const input = document.getElementById(firstErrorFieldId) as HTMLInputElement | null;
        input?.focus();
      }
      return;
    }

    submitBtn.disabled = true;

    try {
      await changePassword(currentEl.value, newEl.value);
      setMsg("Password updated. Redirecting…", "ok");

      window.setTimeout(() => {
        window.location.href = getNextDestination();
      }, 700);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to update password";
      setMsg(message, "error");
      if (message.toLowerCase().includes("current password")) {
        currentEl.focus();
      }
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
