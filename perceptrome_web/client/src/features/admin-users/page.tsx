import { useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useForm } from "react-hook-form";
import {
  adminActivateUser,
  adminCreateUser,
  adminForceResetUser,
  adminListUsers,
  adminResendVerificationUser,
  adminRevokeUserSessions,
  adminSuspendUser,
  adminUpdateUser,
  type AdminUser,
  type AdminUserFilters,
} from "../../auth_api";
import { FormField } from "../../components/FormField";
import { MessageBanner } from "../../components/MessageBanner";
import { EmptyState, ErrorState, LoadingState, StatusBadge } from "../../components/ui/states";
import { randomTempPassword } from "../../lib/utils";
import { adminCreateUserSchema } from "../../lib/validation";

function parseErrors(values: Record<string, unknown>) {
  const parsed = adminCreateUserSchema.safeParse(values);
  if (parsed.success) return {} as Record<string, string>;
  const fieldErrors = parsed.error.flatten().fieldErrors;
  return Object.fromEntries(Object.entries(fieldErrors).map(([key, value]) => [key, value?.[0] ?? ""]));
}

function formatDate(value?: string | null) {
  if (!value) return "—";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

function UserBadges({ user }: { user: AdminUser }) {
  return (
    <div className="cluster">
      <StatusBadge label={user.role} tone={user.role} />
      <StatusBadge label={user.is_active ? "active" : "suspended"} tone={user.is_active ? "success" : "warning"} />
      <StatusBadge label={user.email_verified_at ? "verified" : "pending verification"} tone={user.email_verified_at ? "success" : "warning"} />
      <StatusBadge label={user.is_locked ? `locked (${user.failed_login_count})` : `clear (${user.failed_login_count})`} tone={user.is_locked ? "warning" : "success"} />
      <StatusBadge label={user.must_change_password ? "must change password" : "password current"} tone={user.must_change_password ? "warning" : "success"} />
    </div>
  );
}

function FiltersPanel({ filters, onChange, onReset }: { filters: AdminUserFilters; onChange: (next: AdminUserFilters) => void; onReset: () => void }) {
  return (
    <section className="panel stack">
      <div>
        <p className="eyebrow">Directory controls</p>
        <h2>Search and filters</h2>
        <p className="muted">Filter by role, account state, verification state, and password reset policy.</p>
      </div>
      <div className="filters-grid">
        <label className="input-group"><span className="label">Search</span><input className="input" value={filters.search ?? ""} onChange={(event) => onChange({ ...filters, search: event.target.value })} placeholder="Email, username, or user id" /></label>
        <label className="input-group"><span className="label">Role</span><select className="input" value={filters.role ?? "all"} onChange={(event) => onChange({ ...filters, role: event.target.value as AdminUserFilters["role"] })}><option value="all">All roles</option><option value="user">User</option><option value="admin">Admin</option></select></label>
        <label className="input-group"><span className="label">Account state</span><select className="input" value={filters.state ?? "all"} onChange={(event) => onChange({ ...filters, state: event.target.value as AdminUserFilters["state"] })}><option value="all">All states</option><option value="active">Active</option><option value="suspended">Suspended</option></select></label>
        <label className="input-group"><span className="label">Verification</span><select className="input" value={filters.verification ?? "all"} onChange={(event) => onChange({ ...filters, verification: event.target.value as AdminUserFilters["verification"] })}><option value="all">All users</option><option value="verified">Verified</option><option value="pending">Pending</option></select></label>
        <label className="input-group"><span className="label">Password policy</span><select className="input" value={filters.must_change_password == null ? "all" : filters.must_change_password ? "required" : "current"} onChange={(event) => onChange({ ...filters, must_change_password: event.target.value === "all" ? null : event.target.value === "required" })}><option value="all">All policies</option><option value="required">Must change password</option><option value="current">Password current</option></select></label>
      </div>
      <div className="toolbar"><button className="btn btn--secondary" type="button" onClick={onReset}>Reset filters</button></div>
    </section>
  );
}

function UserTable({ rows, selectedUserId, onSelect, busyUserId, onAction }: { rows: AdminUser[]; selectedUserId?: string; onSelect: (user: AdminUser) => void; busyUserId?: string; onAction: (user: AdminUser, action: "suspend" | "activate" | "force-reset" | "resend-verification" | "revoke-sessions") => void }) {
  return (
    <div className="table-wrap">
      <table className="table admin-users-table">
        <thead><tr><th>User</th><th>Status</th><th>Verification</th><th>Lock</th><th>Last login</th><th>Actions</th></tr></thead>
        <tbody>
          {rows.map((user) => (
            <tr key={user.id} className={selectedUserId === user.id ? "admin-users-table__row admin-users-table__row--selected" : "admin-users-table__row"}>
              <td>
                <button className="admin-users-table__identity" type="button" onClick={() => onSelect(user)}>
                  <strong>{user.email}</strong>
                  <span>{user.username ?? "No username"}</span>
                  <span className="mono">{user.id}</span>
                </button>
              </td>
              <td><StatusBadge label={user.account_state} tone={user.is_active ? "success" : "warning"} /></td>
              <td><StatusBadge label={user.email_verified_at ? "verified" : "pending"} tone={user.email_verified_at ? "success" : "warning"} /></td>
              <td><StatusBadge label={user.is_locked ? `locked (${user.failed_login_count})` : `clear (${user.failed_login_count})`} tone={user.is_locked ? "warning" : "success"} /></td>
              <td className="mono">{formatDate(user.last_login_at)}</td>
              <td>
                <div className="toolbar">
                  <button className="btn btn--secondary btn--sm" type="button" onClick={() => onSelect(user)}>Details</button>
                  <button className="btn btn--secondary btn--sm" type="button" disabled={busyUserId === user.id} onClick={() => onAction(user, user.is_active ? "suspend" : "activate")}>{user.is_active ? "Suspend" : "Activate"}</button>
                  <button className="btn btn--secondary btn--sm" type="button" disabled={busyUserId === user.id} onClick={() => onAction(user, "force-reset")}>Force reset</button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function UserDetailDrawer({ user, busy, onClose, onSave, onAction }: { user: AdminUser; busy: boolean; onClose: () => void; onSave: (input: { username: string; role: "user" | "admin"; is_active: boolean; must_change_password: boolean }) => void; onAction: (action: "suspend" | "activate" | "force-reset" | "resend-verification" | "revoke-sessions") => void }) {
  const [username, setUsername] = useState(user.username ?? "");
  const [role, setRole] = useState<"user" | "admin">((user.role === "admin" ? "admin" : "user"));
  const [isActive, setIsActive] = useState(user.is_active);
  const [mustChangePassword, setMustChangePassword] = useState(user.must_change_password);

  return (
    <aside className="admin-users-drawer panel stack" aria-label="Selected user details">
      <div className="panel-header">
        <div>
          <h2 className="panel-title">User detail</h2>
          <p className="panel-subtitle">Update role, activation, and password policy directly.</p>
        </div>
        <button className="btn btn--secondary btn--sm" type="button" onClick={onClose}>Close</button>
      </div>
      <div className="stack-sm">
        <strong>{user.email}</strong>
        <span className="mono">{user.id}</span>
        <UserBadges user={{ ...user, role, is_active: isActive, must_change_password: mustChangePassword }} />
      </div>
      <div className="definition-grid admin-users-detail-grid">
        <div><span className="label">Created</span><div>{formatDate(user.created_at)}</div></div>
        <div><span className="label">Last login</span><div>{formatDate(user.last_login_at)}</div></div>
        <div><span className="label">Verified at</span><div>{formatDate(user.email_verified_at)}</div></div>
        <div><span className="label">Verification sent</span><div>{formatDate(user.email_verification_sent_at)}</div></div>
        <div><span className="label">Locked until</span><div>{formatDate(user.locked_until)}</div></div>
        <div><span className="label">Failed logins</span><div>{user.failed_login_count}</div></div>
      </div>
      <div className="stack-sm">
        <label className="input-group"><span className="label">Username</span><input className="input" value={username} onChange={(event) => setUsername(event.target.value)} /></label>
        <label className="input-group"><span className="label">Role</span><select className="input" value={role} onChange={(event) => setRole(event.target.value as "user" | "admin")}><option value="user">User</option><option value="admin">Admin</option></select></label>
        <label className="checkbox-label"><input className="input" type="checkbox" checked={isActive} onChange={(event) => setIsActive(event.target.checked)} /><span>Account active</span></label>
        <label className="checkbox-label"><input className="input" type="checkbox" checked={mustChangePassword} onChange={(event) => setMustChangePassword(event.target.checked)} /><span>Must change password</span></label>
      </div>
      <div className="toolbar">
        <button className="btn btn--primary" type="button" disabled={busy} onClick={() => onSave({ username, role, is_active: isActive, must_change_password: mustChangePassword })}>Save changes</button>
      </div>
      <div className="toolbar">
        <button className="btn btn--secondary btn--sm" type="button" disabled={busy} onClick={() => onAction(isActive ? "suspend" : "activate")}>{isActive ? "Suspend" : "Activate"}</button>
        <button className="btn btn--secondary btn--sm" type="button" disabled={busy} onClick={() => onAction("force-reset")}>Force reset</button>
        <button className="btn btn--secondary btn--sm" type="button" disabled={busy} onClick={() => onAction("resend-verification")}>Resend verification</button>
        <button className="btn btn--secondary btn--sm" type="button" disabled={busy} onClick={() => onAction("revoke-sessions")}>Revoke sessions</button>
      </div>
    </aside>
  );
}

export function AdminUsersPage() {
  const [message, setMessage] = useState<string>();
  const [filters, setFilters] = useState<AdminUserFilters>({ role: "all", state: "all", verification: "all", must_change_password: null, search: "" });
  const [selectedUserId, setSelectedUserId] = useState<string>();
  const { register, handleSubmit, setValue, reset, getValues, formState: { isSubmitting } } = useForm<{ email: string; username: string; password: string; role: 'user' | 'admin'; is_active: boolean; must_change_password: boolean }>({ defaultValues: { email: '', username: '', password: randomTempPassword(), role: 'user', is_active: true, must_change_password: true } });
  const errors = parseErrors(getValues());

  const query = useQuery({ queryKey: ["admin-users", filters], queryFn: () => adminListUsers(filters) });
  const rows = query.data?.users ?? [];
  const selectedUser = useMemo(() => rows.find((user) => user.id === selectedUserId) ?? rows[0], [rows, selectedUserId]);
  const selectedResolvedId = selectedUser?.id;

  const mutation = useMutation({
    mutationFn: async ({ user, action }: { user: AdminUser; action: "suspend" | "activate" | "force-reset" | "resend-verification" | "revoke-sessions" }) => {
      if (action === "suspend") return adminSuspendUser(user.id);
      if (action === "activate") return adminActivateUser(user.id);
      if (action === "force-reset") return adminForceResetUser(user.id);
      if (action === "resend-verification") return adminResendVerificationUser(user.id);
      return adminRevokeUserSessions(user.id);
    },
    onSuccess: async (response) => {
      setMessage(response.revoked_session_count > 0 ? `${response.message} (${response.revoked_session_count} sessions revoked).` : response.message);
      setSelectedUserId(response.user.id);
      await query.refetch();
    },
    onError: (error) => setMessage(error instanceof Error ? error.message : "Action failed"),
  });

  return (
    <div className="page-section stack-lg">
      <section className="content-grid content-grid--two">
        <article className="panel stack">
          <div>
            <p className="eyebrow">Admin</p>
            <h1>Users</h1>
            <p className="muted">Manage search, status, verification, role changes, password policy, and session revocation from one page.</p>
          </div>
          <form className="stack" onSubmit={handleSubmit(async (values) => { const parsed = adminCreateUserSchema.safeParse(values); if (!parsed.success) { setMessage(parsed.error.flatten().formErrors[0] ?? 'Please review the form.'); return; } try { const created = await adminCreateUser({ ...values, username: values.username.trim() || null }); setMessage(`Created user: ${created.email}${created.must_change_password ? ' (must change password on first login)' : ''}`); setSelectedUserId(created.id); reset({ email: '', username: '', password: randomTempPassword(), role: 'user', is_active: true, must_change_password: true }); await query.refetch(); } catch (error) { setMessage(error instanceof Error ? error.message : 'Failed to create user'); } })}>
            <div className="row"><FormField label="Email" htmlFor="admin-email" error={errors.email}><input id="admin-email" className="input" type="email" {...register('email')} /></FormField><FormField label="Username (optional)" htmlFor="admin-username" error={errors.username}><input id="admin-username" className="input" type="text" {...register('username')} /></FormField></div>
            <div className="row3"><FormField label="Temporary Password" htmlFor="admin-password" error={errors.password}><input id="admin-password" className="input" type="text" {...register('password')} /></FormField><label className="input-group"><span className="label">Role</span><select className="input" {...register('role')}><option value="user">user</option><option value="admin">admin</option></select><span className="field-error">{errors.role ?? ''}</span></label><div className="cluster"><button className="btn btn--secondary" type="button" onClick={() => setValue('password', randomTempPassword())}>Generate Password</button></div></div>
            <div className="cluster"><label className="checkbox-label"><input className="input" type="checkbox" {...register('is_active')} /><span>Active</span></label><label className="checkbox-label"><input className="input" type="checkbox" {...register('must_change_password')} /><span>Force password change on first login</span></label></div>
            <div className="cluster mt-2"><button className="btn btn--primary" type="submit" disabled={isSubmitting}>Create User</button><button className="btn btn--secondary" type="button" onClick={() => void query.refetch()}>Refresh List</button></div>
            <MessageBanner message={message} tone={message?.startsWith('Created user:') || message?.includes('Revoked') || message === 'User activated' || message === 'User suspended' || message === 'Password reset enforced' || message === 'Verification email resent' ? 'ok' : message ? 'error' : 'plain'} />
          </form>
        </article>

        <FiltersPanel filters={filters} onChange={setFilters} onReset={() => setFilters({ role: "all", state: "all", verification: "all", must_change_password: null, search: "" })} />
      </section>

      <section className="admin-users-layout">
        <article className="panel stack">
          <div className="panel-header">
            <div>
              <h2 className="panel-title">User directory</h2>
              <p className="panel-subtitle">Results from <span className="mono">GET /api/admin/users</span> with server-side filtering.</p>
            </div>
            <StatusBadge label={`${query.data?.total ?? 0} total`} tone="neutral" />
          </div>
          {query.isLoading ? <LoadingState title="Loading users" message="Fetching the latest admin directory." /> : null}
          {query.error instanceof Error ? <ErrorState message={query.error.message} action={<button className="btn btn--secondary" type="button" onClick={() => void query.refetch()}>Retry</button>} /> : null}
          {!query.isLoading && !query.error && rows.length === 0 ? <EmptyState title="No users matched" message="Try widening the search or clearing one of the filters." action={<button className="btn btn--secondary" type="button" onClick={() => setFilters({ role: "all", state: "all", verification: "all", must_change_password: null, search: "" })}>Clear filters</button>} /> : null}
          {!query.isLoading && !query.error && rows.length > 0 ? <UserTable rows={rows} selectedUserId={selectedResolvedId} onSelect={(user) => setSelectedUserId(user.id)} busyUserId={mutation.isPending ? mutation.variables?.user.id : undefined} onAction={(user, action) => mutation.mutate({ user, action })} /> : null}
        </article>

        {selectedUser ? (
          <UserDetailDrawer
            key={selectedUser.id}
            user={selectedUser}
            busy={mutation.isPending}
            onClose={() => setSelectedUserId(undefined)}
            onSave={async (input) => {
              try {
                const updated = await adminUpdateUser(selectedUser.id, { username: input.username.trim() || null, role: input.role, is_active: input.is_active, must_change_password: input.must_change_password });
                setMessage(`Updated ${updated.email}.`);
                setSelectedUserId(updated.id);
                await query.refetch();
              } catch (error) {
                setMessage(error instanceof Error ? error.message : "Failed to update user");
              }
            }}
            onAction={(action) => mutation.mutate({ user: selectedUser, action })}
          />
        ) : (
          <aside className="panel stack">
            <EmptyState title="Select a user" message="Choose a user from the directory to inspect lock state, verification history, and admin actions." />
          </aside>
        )}
      </section>
    </div>
  );
}
