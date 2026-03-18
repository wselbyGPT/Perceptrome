import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { useForm } from "react-hook-form";
import { adminCreateUser, adminListUsers, type AdminUser } from "../../auth_api";
import { AppShell } from "../../app/app-shell";
import { FormField } from "../../components/FormField";
import { MessageBanner } from "../../components/MessageBanner";
import { randomTempPassword } from "../../lib/utils";
import { adminCreateUserSchema } from "../../lib/validation";

function pill(text: string, className: string) {
  return <span className={`badge badge-${className}`}>{text}</span>;
}

function parseErrors(values: Record<string, unknown>) {
  const parsed = adminCreateUserSchema.safeParse(values);
  if (parsed.success) return {} as Record<string, string>;
  const fieldErrors = parsed.error.flatten().fieldErrors;
  return Object.fromEntries(Object.entries(fieldErrors).map(([key, value]) => [key, value?.[0] ?? ""]));
}

function UsersTable({ rows }: { rows: AdminUser[] }) {
  return <div className="table-wrap"><table className="table"><thead><tr><th>Email</th><th>Username</th><th>Role</th><th>Status</th><th>Password Policy</th><th>User ID</th></tr></thead><tbody>{rows.length ? rows.map((user) => <tr key={user.id}><td>{user.email}</td><td>{user.username ?? ''}</td><td>{pill(user.role, user.role === 'admin' ? 'admin' : 'user')}</td><td>{pill(user.is_active ? 'active' : 'inactive', user.is_active ? 'active' : 'inactive')}</td><td>{pill(user.must_change_password ? 'must change' : 'normal', user.must_change_password ? 'force' : 'noforce')}</td><td className="mono">{user.id}</td></tr>) : <tr><td colSpan={6} className="muted">No users found.</td></tr>}</tbody></table></div>;
}

export function AdminUsersPage() {
  const query = useQuery({ queryKey: ["admin-users"], queryFn: adminListUsers });
  const [message, setMessage] = useState<string>();
  const { register, handleSubmit, setValue, reset, getValues, formState: { isSubmitting } } = useForm<{ email: string; username: string; password: string; role: 'user' | 'admin'; is_active: boolean; must_change_password: boolean }>({ defaultValues: { email: '', username: '', password: randomTempPassword(), role: 'user', is_active: true, must_change_password: true } });
  const errors = parseErrors(getValues());
  return <div className="page-admin"><AppShell title="Admin — Users" subtitle="Create and manage Perceptrome users (MVP)" actions={<><Link to="/runs" className="btn btn--secondary">Back to App</Link></>}><div className="admin-wrap"><div className="panel"><h2>Create User</h2><form className="stack" onSubmit={handleSubmit(async (values) => { const parsed = adminCreateUserSchema.safeParse(values); if (!parsed.success) { setMessage(parsed.error.flatten().formErrors[0] ?? 'Please review the form.'); return; } try { const created = await adminCreateUser({ ...values, username: values.username.trim() || null }); setMessage(`Created user: ${created.email}${created.must_change_password ? ' (must change password on first login)' : ''}`); reset({ email: '', username: '', password: randomTempPassword(), role: 'user', is_active: true, must_change_password: true }); await query.refetch(); } catch (error) { setMessage(error instanceof Error ? error.message : 'Failed to create user'); } })}><div className="row"><FormField label="Email" htmlFor="admin-email" error={errors.email}><input id="admin-email" className="input" type="email" {...register('email')} /></FormField><FormField label="Username (optional)" htmlFor="admin-username" error={errors.username}><input id="admin-username" className="input" type="text" {...register('username')} /></FormField></div><div className="row3"><FormField label="Temporary Password" htmlFor="admin-password" error={errors.password}><input id="admin-password" className="input" type="text" {...register('password')} /></FormField><label className="input-group"><span className="label">Role</span><select className="input" {...register('role')}><option value="user">user</option><option value="admin">admin</option></select><span className="field-error">{errors.role ?? ''}</span></label><div className="cluster"><button className="btn btn--secondary" type="button" onClick={() => setValue('password', randomTempPassword())}>Generate Password</button></div></div><div className="cluster"><label className="checkbox-label"><input className="input" type="checkbox" {...register('is_active')} /><span>Active</span></label><label className="checkbox-label"><input className="input" type="checkbox" {...register('must_change_password')} /><span>Force password change on first login</span></label></div><div className="cluster mt-2"><button className="btn btn--primary" type="submit" disabled={isSubmitting}>Create User</button><button className="btn btn--secondary" type="button" onClick={() => void query.refetch()}>Refresh List</button></div><MessageBanner message={message} tone={message?.startsWith('Created user:') ? 'ok' : message ? 'error' : 'plain'} /></form></div><div className="panel"><h2>Users</h2><div className="muted">Admin-only user list from <span className="mono">GET /api/admin/users</span></div><div className={`msg${query.error ? ' error' : query.isLoading ? '' : ' ok'}`}>{query.error instanceof Error ? query.error.message : query.isLoading ? 'Loading users…' : `Loaded ${query.data?.length ?? 0} user(s).`}</div><UsersTable rows={query.data ?? []} /></div></div></AppShell></div>;
}
