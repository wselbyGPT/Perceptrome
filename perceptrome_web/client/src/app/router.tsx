import { Navigate, Route, Routes } from "react-router-dom";
import { RedirectIfAuthenticated, RequireAdmin, RequireAuth } from "./route-guards";
import { LoginPage, ForgotPasswordPage, ResetPasswordPage, VerifyEmailPage } from "../features/auth/pages";
import { RunsPage } from "../features/runs/page";
import { DatasetsPage } from "../features/datasets/page";
import { AdminUsersPage } from "../features/admin-users/page";
import { AuthenticatedLayout } from "../features/layout/authenticated-layout";
import { DashboardPage } from "../features/dashboard/page";
import { ProfilePage } from "../features/profile/page";
import { SecurityPage } from "../features/security/page";
import { AdminInvitationsPage } from "../features/admin-invitations/page";
import { AdminAuditLogPage } from "../features/admin-audit/page";
import { AdminSystemPage } from "../features/admin-system/page";

export function AppRouter() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/dashboard" replace />} />
      <Route path="/login" element={<RedirectIfAuthenticated><LoginPage /></RedirectIfAuthenticated>} />
      <Route path="/forgot-password" element={<RedirectIfAuthenticated><ForgotPasswordPage /></RedirectIfAuthenticated>} />
      <Route path="/reset-password" element={<RedirectIfAuthenticated><ResetPasswordPage /></RedirectIfAuthenticated>} />
      <Route path="/verify-email" element={<RedirectIfAuthenticated><VerifyEmailPage /></RedirectIfAuthenticated>} />

      <Route path="/" element={<RequireAuth><AuthenticatedLayout /></RequireAuth>}>
        <Route path="dashboard" element={<DashboardPage />} />
        <Route path="runs" element={<RunsPage />} />
        <Route path="datasets" element={<DatasetsPage />} />
        <Route path="profile" element={<ProfilePage />} />
        <Route path="security" element={<SecurityPage />} />
        <Route path="admin/users" element={<RequireAdmin><AdminUsersPage /></RequireAdmin>} />
        <Route path="admin/invitations" element={<RequireAdmin><AdminInvitationsPage /></RequireAdmin>} />
        <Route path="admin/audit-log" element={<RequireAdmin><AdminAuditLogPage /></RequireAdmin>} />
        <Route path="admin/system" element={<RequireAdmin><AdminSystemPage /></RequireAdmin>} />
      </Route>
    </Routes>
  );
}
