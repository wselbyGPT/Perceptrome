import { FeedbackNotice, StatusBadge } from "../../components/ui/states";

const services = [
  { name: 'API', status: 'connected', detail: 'REST endpoints reachable through same-origin proxy.' },
  { name: 'WebSocket', status: 'connected', detail: 'Runs page keeps the live channel open while authenticated.' },
  { name: 'Datasets', status: 'active', detail: 'Catalog is available through the routed Datasets page.' },
];

export function AdminSystemPage() {
  return (
    <div className="page-section stack-lg">
      <section className="panel stack">
        <div>
          <p className="eyebrow">Admin</p>
          <h1>System</h1>
          <p className="muted">Operational summaries and future admin controls can live here without leaving the shell.</p>
        </div>
        <FeedbackNotice title="System overview" message="This page currently focuses on presentational readiness while backend admin system endpoints are still maturing." tone="warning" />
        <div className="quick-links">
          {services.map((service) => (
            <article key={service.name} className="quick-link-card quick-link-card--static">
              <div className="cluster"><strong>{service.name}</strong><StatusBadge label={service.status} /></div>
              <span>{service.detail}</span>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
