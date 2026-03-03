import { forgotPassword } from "./auth_api";

const form = document.getElementById("forgot-password-form") as HTMLFormElement | null;
const emailEl = document.getElementById("email") as HTMLInputElement | null;
const submitBtn = document.getElementById("submit-btn") as HTMLButtonElement | null;
const msgEl = document.getElementById("msg") as HTMLDivElement | null;

if (!form || !emailEl || !submitBtn || !msgEl) {
  throw new Error("Forgot password form elements not found");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  msgEl.textContent = "";
  submitBtn.disabled = true;

  try {
    const message = await forgotPassword(emailEl.value.trim());
    msgEl.textContent = message;
  } catch (err) {
    msgEl.textContent = err instanceof Error ? err.message : "Could not submit forgot password request";
  } finally {
    submitBtn.disabled = false;
  }
});
