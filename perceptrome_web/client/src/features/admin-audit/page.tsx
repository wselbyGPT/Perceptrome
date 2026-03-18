import { StatusBadge } from "../../components/ui/states";

const auditEvents = [
  { id: 'evt_001', actor: 'admin@perceptrome.local', action: 'user.create', target: 'scientist@perceptrome.local', result: 'success', at: '2026-03-18T08:45:12Z' },
  { id: 'evt_002', actor: 'admin@perceptrome.local', action: 'auth.password_reset', target: 'researcher@perceptrome.local', result: 'success', at: '2026-03-18T07:11:45Z' },
  { id: 'evt_003', actor: 'ops@perceptrome.local', action: 'system.config_reload', target: 'runtime', result: 'warning', at: '2026-03-17T21:06:33Z' },
];

export function AdminAuditLogPage() {
  return (
    <div className="page-section stack-lg">
      <section className="panel stack">
        <div>
          <p className="eyebrow">Admin</p>
          <h1>Audit log</h1>
          <p className="muted">Shared shell routing is in place for compliance-oriented admin navigation.</p>
        </div>
        <div className="table-wrap">
          <table className="table">
            <thead><tr><th>Event</th><th>Actor</th><th>Target</th><th>Result</th><th>Timestamp</th></tr></thead>
            <tbody>
              {auditEvents.map((event) => (
                <tr key={event.id}>
                  <td><div className="stack-sm"><strong>{event.action}</strong><span className="mono muted">{event.id}</span></div></td>
                  <td>{event.actor}</td>
                  <td>{event.target}</td>
                  <td><StatusBadge label={event.result} tone={event.result === 'success' ? 'success' : 'warning'} /></td>
                  <td className="mono">{event.at}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
