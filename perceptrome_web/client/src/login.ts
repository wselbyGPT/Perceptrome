// client/src/login.ts
import { login } from "./auth_api";

const form = document.getElementById("login-form") as HTMLFormElement | null;
const emailEl = document.getElementById("email") as HTMLInputElement | null;
const passwordEl = document.getElementById("password") as HTMLInputElement | null;
const submitBtn = document.getElementById("submit-btn") as HTMLButtonElement | null;
const msgEl = document.getElementById("msg") as HTMLDivElement | null;

if (!form || !emailEl || !passwordEl || !submitBtn || !msgEl) {
  throw new Error("Login form elements not found");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  msgEl.textContent = "";
  submitBtn.disabled = true;

  try {
    await login(emailEl.value.trim(), passwordEl.value);
    const next = new URLSearchParams(window.location.search).get("next") || "/";
    window.location.href = next;
  } catch (err) {
    msgEl.textContent = err instanceof Error ? err.message : "Login failed";
  } finally {
    submitBtn.disabled = false;
  }
});
