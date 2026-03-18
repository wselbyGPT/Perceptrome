import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useForm } from "react-hook-form";
import { adminCreateUser, adminListUsers, type AdminUser } from "../../auth_api";
import { FormField } from "../../components/FormField";
import { MessageBanner } from "../../components/MessageBanner";
import { StatusBadge } from "../../components/ui/states";
import { randomTempPassword } from "../../lib/utils";
import { adminCreateUserSchema } from "../../lib/validation";

function parseErrors(values: Record<string, unknown>) {
  const parsed = adminCreateUserSchema.safeParse(values);
  if (parsed.success) return {} as Record<string, string>;
  const fieldErrors = parsed.error.flatten().fieldErrors;
  return Object.fromEntries(Object.entries(fieldErrors).map(([key, value]) => [key, value?.[0] ?? ""]));
}

function UsersTable({ rows }: { rows: AdminUser[] }) {
  return (
    <div className="table-wrap">
      <table className="table">
        <thead><tr><th>Email</th><th>Username</th><th>Role</th><th>Status</th><th>Verification</th><th>Lockout</th><th>Password Policy</th><th>Last Login</th><th>User ID</th></tr></thead>
        <tbody>
          {rows.length ? rows.map((user) => (
            <tr key={user.id}>
              <td>{user.email}</td>
              <td>{user.username ?? ''}</td>
              <td><StatusBadge label={user.role} tone={user.role} /></td>
              <td><StatusBadge label={user.account_state} tone={user.is_active ? 'success' : 'warning'} /></td>
              <td><StatusBadge label={user.email_verified_at ? 'verified' : 'pending'} tone={user.email_verified_at ? 'success' : 'warning'} /></td>
              <td><StatusBadge label={user.is_locked ? `locked (${user.failed_login_count})` : `clear (${user.failed_login_count})`} tone={user.is_locked ? 'warning' : 'success'} /></td>
              <td><StatusBadge label={user.must_change_password ? 'must change' : 'current'} tone={user.must_change_password ? 'warning' : 'success'} /></td>
              <td className="mono">{user.last_login_at ?? '—'}</td>
              <td className="mono">{user.id}</td>
            </tr>
          )) : <tr><td colSpan={9} className="muted">No users found.</td></tr>}
        </tbody>
      </table>
    </div>
  );
}

export function AdminUsersPage() {
  const query = useQuery({ queryKey: ["admin-users"], queryFn: adminListUsers });
  const [message, setMessage] = useState<string>();
  const { register, handleSubmit, setValue, reset, getValues, formState: { isSubmitting } } = useForm<{ email: string; username: string; password: string; role: 'user' | 'admin'; is_active: boolean; must_change_password: boolean }>({ defaultValues: { email: '', username: '', password: randomTempPassword(), role: 'user', is_active: true, must_change_password: true } });
  const errors = parseErrors(getValues());

  return (
    <div className="page-section stack-lg">
      <section className="content-grid content-grid--two">
        <article className="panel stack">
          <div>
            <p className="eyebrow">Admin</p>
            <h1>Users</h1>
            <p className="muted">Create and manage Perceptrome users from the shared shell.</p>
          </div>
          <form className="stack" onSubmit={handleSubmit(async (values) => { const parsed = adminCreateUserSchema.safeParse(values); if (!parsed.success) { setMessage(parsed.error.flatten().formErrors[0] ?? 'Please review the form.'); return; } try { const created = await adminCreateUser({ ...values, username: values.username.trim() || null }); setMessage(`Created user: ${created.email}${created.must_change_password ? ' (must change password on first login)' : ''}`); reset({ email: '', username: '', password: randomTempPassword(), role: 'user', is_active: true, must_change_password: true }); await query.refetch(); } catch (error) { setMessage(error instanceof Error ? error.message : 'Failed to create user'); } })}>
            <div className="row"><FormField label="Email" htmlFor="admin-email" error={errors.email}><input id="admin-email" className="input" type="email" {...register('email')} /></FormField><FormField label="Username (optional)" htmlFor="admin-username" error={errors.username}><input id="admin-username" className="input" type="text" {...register('username')} /></FormField></div>
            <div className="row3"><FormField label="Temporary Password" htmlFor="admin-password" error={errors.password}><input id="admin-password" className="input" type="text" {...register('password')} /></FormField><label className="input-group"><span className="label">Role</span><select className="input" {...register('role')}><option value="user">user</option><option value="admin">admin</option></select><span className="field-error">{errors.role ?? ''}</span></label><div className="cluster"><button className="btn btn--secondary" type="button" onClick={() => setValue('password', randomTempPassword())}>Generate Password</button></div></div>
            <div className="cluster"><label className="checkbox-label"><input className="input" type="checkbox" {...register('is_active')} /><span>Active</span></label><label className="checkbox-label"><input className="input" type="checkbox" {...register('must_change_password')} /><span>Force password change on first login</span></label></div>
            <div className="cluster mt-2"><button className="btn btn--primary" type="submit" disabled={isSubmitting}>Create User</button><button className="btn btn--secondary" type="button" onClick={() => void query.refetch()}>Refresh List</button></div>
            <MessageBanner message={message} tone={message?.startsWith('Created user:') ? 'ok' : message ? 'error' : 'plain'} />
          </form>
        </article>

        <article className="panel stack">
          <div>
            <h2>User directory</h2>
            <p className="muted">Admin-only data from <span className="mono">GET /api/admin/users</span>.</p>
          </div>
          <div className={`msg${query.error ? ' error' : query.isLoading ? '' : ' ok'}`}>{query.error instanceof Error ? query.error.message : query.isLoading ? 'Loading users…' : `Loaded ${query.data?.length ?? 0} user(s).`}</div>
          <UsersTable rows={query.data ?? []} />
        </article>
      </section>
    </div>
  );
}
