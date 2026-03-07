import "./styles/index.css";
import { resendVerification, verifyEmail } from "./auth_api";

const tokenInput = document.getElementById("token") as HTMLInputElement | null;
const verifyBtn = document.getElementById("verify-btn") as HTMLButtonElement | null;
const verifyMsg = document.getElementById("verify-msg") as HTMLDivElement | null;
const emailInput = document.getElementById("email") as HTMLInputElement | null;
const resendBtn = document.getElementById("resend-btn") as HTMLButtonElement | null;
const resendMsg = document.getElementById("resend-msg") as HTMLDivElement | null;

if (!tokenInput || !verifyBtn || !verifyMsg || !emailInput || !resendBtn || !resendMsg) {
  throw new Error("Verification form elements not found");
}

const params = new URLSearchParams(window.location.search);
const token = params.get("token");
const email = params.get("email");
if (token) tokenInput.value = token;
if (email) emailInput.value = email;

verifyBtn.addEventListener("click", async () => {
  verifyMsg.textContent = "";
  verifyBtn.disabled = true;
  try {
    const message = await verifyEmail(tokenInput.value.trim());
    verifyMsg.textContent = `${message}. You can now return to login.`;
  } catch (err) {
    verifyMsg.textContent = err instanceof Error ? err.message : "Verification failed";
  } finally {
    verifyBtn.disabled = false;
  }
});

resendBtn.addEventListener("click", async () => {
  resendMsg.textContent = "";
  resendBtn.disabled = true;
  try {
    const message = await resendVerification(emailInput.value.trim());
    resendMsg.textContent = message;
  } catch (err) {
    resendMsg.textContent = err instanceof Error ? err.message : "Could not resend email";
  } finally {
    resendBtn.disabled = false;
  }
});
