import { useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { adminListAuditEvents, type AuditEvent, type AuditFilters } from "../../auth_api";
import { EmptyState, ErrorState, LoadingState, StatusBadge } from "../../components/ui/states";

function formatDate(value?: string | null) {
  if (!value) return "—";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

function metadataDisplay(metadata: Record<string, unknown>) {
  const entries = Object.entries(metadata);
  if (entries.length === 0) return <span className="muted">No metadata</span>;
  return <pre className="admin-audit-metadata">{JSON.stringify(metadata, null, 2)}</pre>;
}

function AuditFiltersPanel({ filters, onChange, onReset }: { filters: AuditFilters; onChange: (next: AuditFilters) => void; onReset: () => void }) {
  return (
    <section className="panel stack">
      <div>
        <p className="eyebrow">Audit filters</p>
        <h2>Compliance search</h2>
        <p className="muted">Filter by action, actor, target, or a free-text metadata search.</p>
      </div>
      <div className="filters-grid">
        <label className="input-group"><span className="label">Action</span><input className="input" value={filters.action ?? ""} onChange={(event) => onChange({ ...filters, action: event.target.value })} placeholder="admin.user_created" /></label>
        <label className="input-group"><span className="label">Actor</span><input className="input" value={filters.actor ?? ""} onChange={(event) => onChange({ ...filters, actor: event.target.value })} placeholder="admin@example.com or id" /></label>
        <label className="input-group"><span className="label">Target</span><input className="input" value={filters.target ?? ""} onChange={(event) => onChange({ ...filters, target: event.target.value })} placeholder="member@example.com or id" /></label>
        <label className="input-group"><span className="label">Search metadata</span><input className="input" value={filters.search ?? ""} onChange={(event) => onChange({ ...filters, search: event.target.value })} placeholder="role, invitation id, session count" /></label>
      </div>
      <div className="toolbar"><button className="btn btn--secondary" type="button" onClick={onReset}>Reset filters</button></div>
    </section>
  );
}

function AuditTable({ rows }: { rows: AuditEvent[] }) {
  return (
    <div className="table-wrap">
      <table className="table">
        <thead><tr><th>Timestamp</th><th>Action</th><th>Actor</th><th>Target</th><th>Metadata</th></tr></thead>
        <tbody>
          {rows.map((event) => (
            <tr key={event.id}>
              <td className="mono">{formatDate(event.created_at)}</td>
              <td><div className="stack-sm"><strong>{event.action}</strong><span className="mono muted">{event.id}</span></div></td>
              <td><div className="stack-sm"><span>{event.actor_email ?? "System"}</span><span className="mono muted">{event.actor_user_id ?? "—"}</span></div></td>
              <td><div className="stack-sm"><span>{event.target_email ?? "—"}</span><span className="mono muted">{event.target_user_id ?? "—"}</span></div></td>
              <td>
                <div className="stack-sm">
                  {metadataDisplay(event.metadata)}
                  <div className="cluster muted"><span>IP: {event.ip_address ?? "—"}</span><span>UA: {event.user_agent ?? "—"}</span></div>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function AdminAuditLogPage() {
  const [filters, setFilters] = useState<AuditFilters>({ action: "", actor: "", target: "", search: "" });
  const query = useQuery({ queryKey: ["admin-audit", filters], queryFn: () => adminListAuditEvents(filters) });
  const rows = query.data?.events ?? [];

  return (
    <div className="page-section stack-lg">
      <AuditFiltersPanel filters={filters} onChange={setFilters} onReset={() => setFilters({ action: "", actor: "", target: "", search: "" })} />
      <section className="panel stack">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Admin</p>
            <h1>Audit log</h1>
            <p className="muted">Review actor, target, action, and metadata for security-sensitive admin operations with exact timestamps.</p>
          </div>
          <StatusBadge label={`${query.data?.total ?? 0} total`} tone="neutral" />
        </div>
        {query.isLoading ? <LoadingState title="Loading audit log" message="Querying the latest compliance events." /> : null}
        {query.error instanceof Error ? <ErrorState message={query.error.message} action={<button className="btn btn--secondary" type="button" onClick={() => void query.refetch()}>Retry</button>} /> : null}
        {!query.isLoading && !query.error && rows.length === 0 ? <EmptyState title="No audit events matched" message="Adjust filters to widen the compliance search." /> : null}
        {!query.isLoading && !query.error && rows.length > 0 ? <AuditTable rows={rows} /> : null}
      </section>
    </div>
  );
}
