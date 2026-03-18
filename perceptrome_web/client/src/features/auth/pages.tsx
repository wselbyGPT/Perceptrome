import { useMemo, useState, type PropsWithChildren } from "react";
import { useForm } from "react-hook-form";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { forgotPassword, resendVerification, resetPassword, verifyEmail } from "../../auth_api";
import { useAuth } from "./auth-context";
import { FormField } from "../../components/FormField";
import { MessageBanner } from "../../components/MessageBanner";
import { FeedbackNotice, StatusBadge } from "../../components/ui/states";
import {
  changePasswordSchema,
  forgotPasswordSchema,
  loginSchema,
  resendVerificationSchema,
  resetPasswordSchema,
  verifyEmailSchema,
} from "../../lib/validation";

function parseErrors<T extends Record<string, unknown>>(schema: { safeParse: (input: unknown) => { success: boolean; error?: { flatten: () => { fieldErrors: Record<string, string[]> } } } }, values: T) {
  const result = schema.safeParse(values);
  if (result.success) return {} as Record<string, string>;
  const fieldErrors = result.error?.flatten().fieldErrors ?? {};
  return Object.fromEntries(Object.entries(fieldErrors).map(([key, value]) => [key, value?.[0] ?? ""]));
}

function AuthCard({ title, subtitle, children }: PropsWithChildren<{ title: string; subtitle: string }>) {
  return (
    <div className="auth-page">
      <div className="panel auth-card">
        <p className="eyebrow">Perceptrome</p>
        <h1>{title}</h1>
        <p className="auth-sub">{subtitle}</p>
        {children}
      </div>
    </div>
  );
}

export function LoginPage() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const { loginWithPassword } = useAuth();
  const [message, setMessage] = useState<string>();
  const [verifyEmailAddress, setVerifyEmailAddress] = useState<string>();
  const { register, handleSubmit, formState: { isSubmitting }, getValues } = useForm<{ email: string; password: string }>({ defaultValues: { email: "", password: "" } });
  const errors = parseErrors(loginSchema, getValues());

  return (
    <AuthCard title="Login" subtitle="Sign in to access runs, datasets, and administrative controls through the new SPA shell.">
      <form
        className="stack"
        onSubmit={handleSubmit(async (values) => {
          setMessage(undefined);
          setVerifyEmailAddress(undefined);
          const parsed = loginSchema.safeParse(values);
          if (!parsed.success) {
            setMessage(parsed.error.flatten().formErrors[0] ?? "Enter your credentials.");
            return;
          }
          try {
            await loginWithPassword({ email: values.email.trim(), password: values.password });
            navigate(params.get("next") || "/dashboard", { replace: true });
          } catch (error) {
            const nextMessage = error instanceof Error ? error.message : "Login failed";
            setMessage(nextMessage);
            if (nextMessage === "Email verification required") {
              setVerifyEmailAddress(values.email.trim());
            }
          }
        })}
      >
        <FormField label="Email" htmlFor="email" error={errors.email}>
          <input id="email" className="input" type="email" autoComplete="username" {...register("email")} />
        </FormField>
        <FormField label="Password" htmlFor="password" error={errors.password}>
          <input id="password" className="input" type="password" autoComplete="current-password" {...register("password")} />
        </FormField>
        <button className="btn btn--primary" type="submit" disabled={isSubmitting}>Sign in</button>
        <MessageBanner message={message} tone="error" />
        {verifyEmailAddress ? (
          <FeedbackNotice title="Email verification required" message="This account must verify its email address before signing in." tone="warning" actions={<Link to={`/verify-email?email=${encodeURIComponent(verifyEmailAddress)}`} className="btn btn--secondary btn--sm">Verify email</Link>} />
        ) : null}
        <div className="auth-links">
          <Link to="/forgot-password">Forgot password?</Link>
          <Link to="/verify-email">Verify email</Link>
        </div>
      </form>
    </AuthCard>
  );
}

export function ForgotPasswordPage() {
  const [message, setMessage] = useState<string>();
  const { register, handleSubmit, formState: { isSubmitting }, getValues } = useForm<{ email: string }>({ defaultValues: { email: "" } });
  const errors = parseErrors(forgotPasswordSchema, getValues());
  return (
    <AuthCard title="Forgot Password" subtitle="Request a password reset token for your account.">
      <form className="stack" onSubmit={handleSubmit(async (values) => { const parsed = forgotPasswordSchema.safeParse(values); if (!parsed.success) { setMessage(parsed.error.flatten().formErrors[0] ?? "Enter a valid email address"); return; } try { setMessage(await forgotPassword(values.email.trim())); } catch (error) { setMessage(error instanceof Error ? error.message : "Could not submit forgot password request"); } })}>
        <FormField label="Email" htmlFor="forgot-email" error={errors.email}><input id="forgot-email" className="input" type="email" {...register("email")} /></FormField>
        <button className="btn btn--primary" type="submit" disabled={isSubmitting}>Send reset email</button>
        <MessageBanner message={message} tone={message?.toLowerCase().includes("could not") ? "error" : "plain"} />
        <div className="auth-links"><Link to="/login">Back to login</Link></div>
      </form>
    </AuthCard>
  );
}

export function ResetPasswordPage() {
  const [params] = useSearchParams();
  const token = params.get("token") ?? "";
  const [message, setMessage] = useState<string>();
  const { register, handleSubmit, formState: { isSubmitting }, getValues } = useForm<{ token: string; newPassword: string; confirmPassword: string }>({ defaultValues: { token, newPassword: "", confirmPassword: "" } });
  const errors = parseErrors(resetPasswordSchema, getValues());
  return (
    <AuthCard title="Reset Password" subtitle="Use the emailed reset token to choose a new password.">
      <form className="stack" onSubmit={handleSubmit(async (values) => { const parsed = resetPasswordSchema.safeParse(values); if (!parsed.success) { setMessage(parsed.error.flatten().formErrors[0] ?? "Please review the form."); return; } try { const response = await resetPassword(values.token.trim(), values.newPassword); setMessage(`${response}. You can now return to login.`); } catch (error) { setMessage(error instanceof Error ? error.message : "Could not reset password"); } })}>
        <FormField label="Reset token" htmlFor="token" error={errors.token}><input id="token" className="input" type="text" {...register("token")} /></FormField>
        <FormField label="New password" htmlFor="new-password" error={errors.newPassword}><input id="new-password" className="input" type="password" {...register("newPassword")} /></FormField>
        <FormField label="Confirm new password" htmlFor="confirm-password" error={errors.confirmPassword}><input id="confirm-password" className="input" type="password" {...register("confirmPassword")} /></FormField>
        <button className="btn btn--primary" type="submit" disabled={isSubmitting}>Reset password</button>
        <MessageBanner message={message} tone={message?.includes("return to login") ? "ok" : "plain"} />
        <div className="auth-links"><Link to="/login">Back to login</Link></div>
      </form>
    </AuthCard>
  );
}

export function VerifyEmailPage() {
  const [params] = useSearchParams();
  const [verifyMessage, setVerifyMessage] = useState<string>();
  const [resendMessage, setResendMessage] = useState<string>();
  const verifyForm = useForm<{ token: string }>({ defaultValues: { token: params.get("token") ?? "" } });
  const resendForm = useForm<{ email: string }>({ defaultValues: { email: params.get("email") ?? "" } });
  const verifyErrors = parseErrors(verifyEmailSchema, verifyForm.getValues());
  const resendErrors = parseErrors(resendVerificationSchema, resendForm.getValues());
  return (
    <AuthCard title="Verify Email" subtitle="Confirm your email token or request a fresh verification message.">
      <div className="stack-lg">
        <form className="stack" onSubmit={verifyForm.handleSubmit(async (values) => { const parsed = verifyEmailSchema.safeParse(values); if (!parsed.success) { setVerifyMessage(parsed.error.flatten().formErrors[0] ?? "Verification token is required"); return; } try { setVerifyMessage(`${await verifyEmail(values.token.trim())}. You can now return to login.`); } catch (error) { setVerifyMessage(error instanceof Error ? error.message : "Verification failed"); } })}>
          <FormField label="Verification token" htmlFor="verify-token" error={verifyErrors.token}><input id="verify-token" className="input" type="text" {...verifyForm.register("token")} /></FormField>
          <button className="btn btn--primary" type="submit" disabled={verifyForm.formState.isSubmitting}>Verify email</button>
          <MessageBanner message={verifyMessage} tone={verifyMessage?.includes("return to login") ? "ok" : "plain"} />
        </form>
        <form className="stack" onSubmit={resendForm.handleSubmit(async (values) => { const parsed = resendVerificationSchema.safeParse(values); if (!parsed.success) { setResendMessage(parsed.error.flatten().formErrors[0] ?? "Enter your email"); return; } try { setResendMessage(await resendVerification(values.email.trim())); } catch (error) { setResendMessage(error instanceof Error ? error.message : "Could not resend email"); } })}>
          <FormField label="Email" htmlFor="resend-email" error={resendErrors.email}><input id="resend-email" className="input" type="email" {...resendForm.register("email")} /></FormField>
          <button className="btn btn--secondary" type="submit" disabled={resendForm.formState.isSubmitting}>Resend verification</button>
          <MessageBanner message={resendMessage} />
        </form>
        <div className="auth-links"><Link to="/login">Back to login</Link></div>
      </div>
    </AuthCard>
  );
}

function scorePasswordStrength(password: string): { score: number; label: string } {
  if (!password) return { score: 0, label: "Too weak" };
  let score = 0;
  if (password.length >= 12) score += 1;
  if (/[A-Z]/.test(password)) score += 1;
  if (/[a-z]/.test(password)) score += 1;
  if (/\d/.test(password)) score += 1;
  if (/[^A-Za-z0-9]/.test(password)) score += 1;
  if (score <= 2) return { score, label: "Weak" };
  if (score <= 4) return { score, label: "Good" };
  return { score, label: "Strong" };
}

export function ChangePasswordPage() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const { me, logoutAndClear, changePasswordAndRefresh } = useAuth();
  const [message, setMessage] = useState<string>();
  const [visible, setVisible] = useState(false);
  const { register, handleSubmit, watch, formState: { isSubmitting }, getValues } = useForm<{ currentPassword: string; newPassword: string; confirmPassword: string }>({ defaultValues: { currentPassword: "", newPassword: "", confirmPassword: "" } });
  const newPassword = watch("newPassword");
  const strength = useMemo(() => scorePasswordStrength(newPassword), [newPassword]);
  const errors = parseErrors(changePasswordSchema, getValues());

  return (
    <div className="page-section stack-lg">
      <section className="panel stack">
        <div>
          <p className="eyebrow">Security</p>
          <h1>{me?.must_change_password ? "Password change required" : "Security settings"}</h1>
          <p className="muted">Rotate credentials and confirm account protection posture.</p>
        </div>
        <div className="cluster">
          <StatusBadge label={me?.must_change_password ? 'action required' : 'healthy'} tone={me?.must_change_password ? 'warning' : 'success'} />
          <StatusBadge label={me?.email_verified_at ? 'verified email' : 'pending verification'} tone={me?.email_verified_at ? 'success' : 'warning'} />
        </div>
        <form className="stack" onSubmit={handleSubmit(async (values) => { const parsed = changePasswordSchema.safeParse(values); if (!parsed.success) { setMessage("Please fix the highlighted password issues."); return; } try { await changePasswordAndRefresh(values.currentPassword, values.newPassword); setMessage("Password updated. Redirecting…"); window.setTimeout(() => { navigate(params.get("next") || "/dashboard", { replace: true }); }, 700); } catch (error) { setMessage(error instanceof Error ? error.message : "Failed to update password"); } })}>
          <FormField label="Current Password" htmlFor="current-password" error={errors.currentPassword}><input id="current-password" className="input" type={visible ? "text" : "password"} autoComplete="current-password" {...register("currentPassword")} /></FormField>
          <FormField label="New Password" htmlFor="change-new-password" error={errors.newPassword}><input id="change-new-password" className="input" type={visible ? "text" : "password"} autoComplete="new-password" {...register("newPassword")} /></FormField>
          <div className="strength" aria-live="polite"><progress max={5} value={strength.score}></progress><div className="muted">Strength: {strength.label}</div></div>
          <div className="input-group"><span className="label">Confirm New Password</span><div className="password-row"><input id="confirm-password-change" className="input" type={visible ? "text" : "password"} autoComplete="new-password" {...register("confirmPassword")} /><button className="btn btn--secondary btn--sm" type="button" onClick={() => setVisible((value) => !value)}>{visible ? "Hide" : "Show"}</button></div><span className="field-error">{errors.confirmPassword ?? ""}</span></div>
          <div className="cluster mt-2"><button className="btn btn--primary" type="submit" disabled={isSubmitting}>Update Password</button><button className="btn btn--secondary" type="button" onClick={async () => { await logoutAndClear(); navigate('/login'); }}>Logout</button></div>
          <MessageBanner message={message} tone={message?.startsWith("Password updated") ? "ok" : message ? "error" : "plain"} />
        </form>
      </section>
    </div>
  );
}
