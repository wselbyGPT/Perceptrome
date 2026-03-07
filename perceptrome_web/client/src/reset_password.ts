import "./styles/index.css";
import { resetPassword } from "./auth_api";

const form = document.getElementById("reset-password-form") as HTMLFormElement | null;
const tokenEl = document.getElementById("token") as HTMLInputElement | null;
const newPasswordEl = document.getElementById("new-password") as HTMLInputElement | null;
const confirmPasswordEl = document.getElementById("confirm-password") as HTMLInputElement | null;
const submitBtn = document.getElementById("submit-btn") as HTMLButtonElement | null;
const msgEl = document.getElementById("msg") as HTMLDivElement | null;

if (!form || !tokenEl || !newPasswordEl || !confirmPasswordEl || !submitBtn || !msgEl) {
  throw new Error("Reset password form elements not found");
}

const tokenFromQuery = new URLSearchParams(window.location.search).get("token");
if (tokenFromQuery) tokenEl.value = tokenFromQuery;

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  msgEl.textContent = "";
  submitBtn.disabled = true;

  try {
    const token = tokenEl.value.trim();
    const newPassword = newPasswordEl.value;
    const confirmPassword = confirmPasswordEl.value;

    if (newPassword.length < 8) {
      throw new Error("New password must be at least 8 characters");
    }
    if (newPassword !== confirmPassword) {
      throw new Error("New passwords do not match");
    }

    const message = await resetPassword(token, newPassword);
    msgEl.textContent = `${message}. You can now return to login.`;
  } catch (err) {
    msgEl.textContent = err instanceof Error ? err.message : "Could not reset password";
  } finally {
    submitBtn.disabled = false;
  }
});
