import { Navigate, Route, Routes } from "react-router-dom";
import { RedirectIfAuthenticated, RequireAdmin, RequireAuth } from "./route-guards";
import { LoginPage, ForgotPasswordPage, ResetPasswordPage, VerifyEmailPage, ChangePasswordPage } from "../features/auth/pages";
import { RunsPage } from "../features/runs/page";
import { DatasetsPage } from "../features/datasets/page";
import { AdminUsersPage } from "../features/admin-users/page";
import { RunsWebSocketProvider } from "../features/runs/ws-provider";

export function AppRouter() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/runs" replace />} />
      <Route path="/login" element={<RedirectIfAuthenticated><LoginPage /></RedirectIfAuthenticated>} />
      <Route path="/forgot-password" element={<RedirectIfAuthenticated><ForgotPasswordPage /></RedirectIfAuthenticated>} />
      <Route path="/reset-password" element={<RedirectIfAuthenticated><ResetPasswordPage /></RedirectIfAuthenticated>} />
      <Route path="/verify-email" element={<RedirectIfAuthenticated><VerifyEmailPage /></RedirectIfAuthenticated>} />
      <Route path="/profile/change-password" element={<RequireAuth><ChangePasswordPage /></RequireAuth>} />
      <Route path="/runs" element={<RequireAuth><RunsWebSocketProvider><RunsPage /></RunsWebSocketProvider></RequireAuth>} />
      <Route path="/datasets" element={<RequireAuth><DatasetsPage /></RequireAuth>} />
      <Route path="/admin/users" element={<RequireAuth><RequireAdmin><AdminUsersPage /></RequireAdmin></RequireAuth>} />
    </Routes>
  );
}
