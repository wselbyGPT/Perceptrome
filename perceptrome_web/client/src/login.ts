import "./styles/index.css";
// client/src/login.ts
import { login } from "./auth_api";

const form = document.getElementById("login-form") as HTMLFormElement | null;
const emailEl = document.getElementById("email") as HTMLInputElement | null;
const passwordEl = document.getElementById("password") as HTMLInputElement | null;
const submitBtn = document.getElementById("submit-btn") as HTMLButtonElement | null;
const msgEl = document.getElementById("msg") as HTMLDivElement | null;
const verifyHelpEl = document.getElementById("verify-help") as HTMLDivElement | null;

if (!form || !emailEl || !passwordEl || !submitBtn || !msgEl || !verifyHelpEl) {
  throw new Error("Login form elements not found");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  msgEl.textContent = "";
  verifyHelpEl.innerHTML = "";
  submitBtn.disabled = true;

  try {
    await login(emailEl.value.trim(), passwordEl.value);
    const next = new URLSearchParams(window.location.search).get("next") || "/";
    window.location.href = next;
  } catch (err) {
    const message = err instanceof Error ? err.message : "Login failed";
    msgEl.textContent = message;

    if (message === "Email verification required") {
      const email = encodeURIComponent(emailEl.value.trim());
      verifyHelpEl.innerHTML = `Please verify your email. <a href="/verify_email.html?email=${email}">Open verification page</a>`;
    }
  } finally {
    submitBtn.disabled = false;
  }
});
