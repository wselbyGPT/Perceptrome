import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";

import { adminCreateInvitation, adminListInvitations, adminRevokeInvitation, type AdminInvitation, type AdminInvitationFilters } from "../../auth_api";
import { MessageBanner } from "../../components/MessageBanner";
import { EmptyState, ErrorState, FeedbackNotice, LoadingState, StatusBadge } from "../../components/ui/states";

function formatDate(value?: string | null) {
  if (!value) return "—";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

function statusTone(status: string) {
  if (status === "accepted") return "success";
  if (status === "revoked") return "muted";
  if (status === "expired") return "warning";
  return "info";
}

function InvitationFiltersPanel({ filters, onChange, onReset }: { filters: AdminInvitationFilters; onChange: (next: AdminInvitationFilters) => void; onReset: () => void }) {
  return (
    <section className="panel stack">
      <div>
        <p className="eyebrow">Filters</p>
        <h2>Invitation directory</h2>
        <p className="muted">Inspect pending, accepted, revoked, and expired invitations with server-side filtering.</p>
      </div>
      <div className="filters-grid">
        <label className="input-group"><span className="label">Search</span><input className="input" value={filters.search ?? ""} onChange={(event) => onChange({ ...filters, search: event.target.value })} placeholder="Email or invitation id" /></label>
        <label className="input-group"><span className="label">Role</span><select className="input" value={filters.role ?? "all"} onChange={(event) => onChange({ ...filters, role: event.target.value as AdminInvitationFilters["role"] })}><option value="all">All roles</option><option value="user">User</option><option value="admin">Admin</option></select></label>
        <label className="input-group"><span className="label">Status</span><select className="input" value={filters.status ?? "all"} onChange={(event) => onChange({ ...filters, status: event.target.value as AdminInvitationFilters["status"] })}><option value="all">All statuses</option><option value="pending">Pending</option><option value="accepted">Accepted</option><option value="revoked">Revoked</option><option value="expired">Expired</option></select></label>
      </div>
      <div className="toolbar"><button className="btn btn--secondary" type="button" onClick={onReset}>Reset filters</button></div>
    </section>
  );
}

function InvitationTable({ rows, busyInvitationId, onRevoke }: { rows: AdminInvitation[]; busyInvitationId?: string; onRevoke: (invitation: AdminInvitation) => void }) {
  return (
    <div className="table-wrap">
      <table className="table">
        <thead><tr><th>Email</th><th>Role</th><th>Status</th><th>Created</th><th>Expires</th><th>Details</th><th>Action</th></tr></thead>
        <tbody>
          {rows.map((invite) => (
            <tr key={invite.id}>
              <td><div className="stack-sm"><strong>{invite.email}</strong><span className="mono muted">{invite.id}</span></div></td>
              <td><StatusBadge label={invite.role} tone={invite.role} /></td>
              <td><StatusBadge label={invite.status} tone={statusTone(invite.status)} /></td>
              <td className="mono">{formatDate(invite.created_at)}</td>
              <td className="mono">{formatDate(invite.expires_at)}</td>
              <td>
                <div className="stack-sm">
                  <span>Accepted: {formatDate(invite.accepted_at)}</span>
                  <span>Revoked: {formatDate(invite.revoked_at)}</span>
                </div>
              </td>
              <td>{invite.status === "pending" ? <button className="btn btn--secondary btn--sm" type="button" disabled={busyInvitationId === invite.id} onClick={() => onRevoke(invite)}>Revoke</button> : <span className="muted">—</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function AdminInvitationsPage() {
  const [filters, setFilters] = useState<AdminInvitationFilters>({ role: "all", status: "all", search: "" });
  const [form, setForm] = useState({ email: "", role: "user" as "user" | "admin", reissue: true });
  const [message, setMessage] = useState<string>();
  const [lastCreated, setLastCreated] = useState<AdminInvitation | null>(null);

  const query = useQuery({ queryKey: ["admin-invitations", filters], queryFn: () => adminListInvitations(filters) });

  const createMutation = useMutation({
    mutationFn: () => adminCreateInvitation(form),
    onSuccess: async (invitation) => {
      setLastCreated(invitation);
      setMessage(`Invitation issued for ${invitation.email}.`);
      setForm({ email: "", role: "user", reissue: true });
      await query.refetch();
    },
    onError: (error) => setMessage(error instanceof Error ? error.message : "Failed to create invitation"),
  });

  const revokeMutation = useMutation({
    mutationFn: (invitationId: string) => adminRevokeInvitation(invitationId),
    onSuccess: async ({ invitation }) => {
      setMessage(`Invitation revoked for ${invitation.email}.`);
      await query.refetch();
    },
    onError: (error) => setMessage(error instanceof Error ? error.message : "Failed to revoke invitation"),
  });

  const rows = query.data?.invitations ?? [];

  return (
    <div className="page-section stack-lg">
      <section className="content-grid content-grid--two">
        <article className="panel stack">
          <div>
            <p className="eyebrow">Admin</p>
            <h1>Invitations</h1>
            <p className="muted">Create new invitations, reissue to refresh tokens, and monitor accepted/revoked/expired states with exact timestamps.</p>
          </div>
          <div className="stack-sm">
            <label className="input-group"><span className="label">Email</span><input className="input" type="email" value={form.email} onChange={(event) => setForm((current) => ({ ...current, email: event.target.value }))} placeholder="invitee@example.com" /></label>
            <label className="input-group"><span className="label">Role</span><select className="input" value={form.role} onChange={(event) => setForm((current) => ({ ...current, role: event.target.value as "user" | "admin" }))}><option value="user">user</option><option value="admin">admin</option></select></label>
            <label className="checkbox-label"><input className="input" type="checkbox" checked={form.reissue} onChange={(event) => setForm((current) => ({ ...current, reissue: event.target.checked }))} /><span>Reissue existing pending invitation if one exists</span></label>
            <div className="toolbar"><button className="btn btn--primary" type="button" disabled={createMutation.isPending || !form.email.trim()} onClick={() => createMutation.mutate()}>Create invitation</button><button className="btn btn--secondary" type="button" onClick={() => void query.refetch()}>Refresh list</button></div>
          </div>
          <MessageBanner message={message} tone={message?.includes("issued") || message?.includes("revoked") ? "ok" : message ? "error" : "plain"} />
          {lastCreated ? (
            <FeedbackNotice
              title="Latest raw invitation token"
              message={`Store this token or URL now. It is shown only at creation/reissue time. Token: ${lastCreated.token_preview ?? "unavailable"}`}
              tone="warning"
              actions={lastCreated.invite_url ? <a className="btn btn--secondary btn--sm" href={lastCreated.invite_url} target="_blank" rel="noreferrer">Open invite link</a> : null}
            />
          ) : null}
        </article>
        <InvitationFiltersPanel filters={filters} onChange={setFilters} onReset={() => setFilters({ role: "all", status: "all", search: "" })} />
      </section>

      <section className="panel stack">
        <div className="panel-header">
          <div>
            <h2 className="panel-title">Invitation ledger</h2>
            <p className="panel-subtitle">Server data from <span className="mono">GET /api/admin/invitations</span>.</p>
          </div>
          <StatusBadge label={`${query.data?.total ?? 0} total`} tone="neutral" />
        </div>
        {query.isLoading ? <LoadingState title="Loading invitations" message="Fetching current invitation state and lifecycle timestamps." /> : null}
        {query.error instanceof Error ? <ErrorState message={query.error.message} action={<button className="btn btn--secondary" type="button" onClick={() => void query.refetch()}>Retry</button>} /> : null}
        {!query.isLoading && !query.error && rows.length === 0 ? <EmptyState title="No invitations matched" message="Try widening the filters or create a new invitation." /> : null}
        {!query.isLoading && !query.error && rows.length > 0 ? <InvitationTable rows={rows} busyInvitationId={revokeMutation.isPending ? revokeMutation.variables : undefined} onRevoke={(invitation) => revokeMutation.mutate(invitation.id)} /> : null}
      </section>
    </div>
  );
}
