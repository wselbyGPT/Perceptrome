import { Link } from "react-router-dom";
import { ErrorState, FeedbackNotice, LoadingState, StatusBadge } from "../../components/ui/states";
import { useAuth } from "../auth/auth-context";
import { useDatasetCountQuery } from "../datasets/hooks";
import { useRunSummaryQuery } from "../runs/hooks";

export function DashboardPage() {
  const { me } = useAuth();
  const summaryQuery = useRunSummaryQuery();
  const datasetsQuery = useDatasetCountQuery();

  if (summaryQuery.isLoading || datasetsQuery.isLoading) {
    return <LoadingState title="Loading dashboard" message="Pulling run and dataset summaries." />;
  }

  if (summaryQuery.error) {
    return <ErrorState message={summaryQuery.error instanceof Error ? summaryQuery.error.message : "Failed to load the dashboard."} action={<button className="btn btn--secondary" onClick={() => void summaryQuery.refetch()}>Retry</button>} />;
  }

  const summary = summaryQuery.data;
  const datasets = datasetsQuery.data ?? [];

  if (!summary) {
    return <LoadingState title="Loading dashboard" message="Pulling run and dataset summaries." />;
  }

  return (
    <div className="page-section stack-lg">
      <section className="hero-card panel">
        <div>
          <p className="eyebrow">Workspace overview</p>
          <h1>Welcome back{me ? `, ${me.username ?? me.email}` : ''}.</h1>
          <p className="hero-copy">Monitor active experimentation, jump into dataset curation, and keep account operations close at hand.</p>
        </div>
        <div className="hero-actions">
          <Link to="/runs" className="btn btn--primary">Open Runs</Link>
          <Link to="/datasets" className="btn btn--secondary">Browse Datasets</Link>
        </div>
      </section>

      <section className="stats-grid">
        <article className="panel stat-panel"><span className="label">Queued</span><strong>{summary.queued}</strong><StatusBadge label="queued" /></article>
        <article className="panel stat-panel"><span className="label">Running</span><strong>{summary.running}</strong><StatusBadge label="running" /></article>
        <article className="panel stat-panel"><span className="label">Completed</span><strong>{summary.completed}</strong><StatusBadge label="completed" /></article>
        <article className="panel stat-panel"><span className="label">Datasets</span><strong>{datasets.length}</strong><StatusBadge label="catalog" tone="neutral" /></article>
      </section>

      <section className="content-grid content-grid--two">
        <article className="panel stack">
          <div>
            <h2>Recent system health</h2>
            <p className="muted">Quick signals derived from the current run summary.</p>
          </div>
          <FeedbackNotice
            title={summary.failed > 0 ? "Recent failures detected" : "Runs look healthy"}
            message={summary.failed > 0 ? `${summary.failed} failed run(s) need attention. Use the Runs page to inspect logs and artifacts.` : "No failed runs are reported in the latest summary snapshot."}
            tone={summary.failed > 0 ? 'warning' : 'success'}
            actions={<Link to="/runs" className="btn btn--secondary btn--sm">Review runs</Link>}
          />
          <div className="stack-sm">
            <div className="kv-row"><span>Total runs</span><strong>{summary.total_runs}</strong></div>
            <div className="kv-row"><span>Latest failed run</span><strong className="mono">{summary.latest_failed_run_id ?? 'n/a'}</strong></div>
            <div className="kv-row"><span>Latest failure timestamp</span><strong>{summary.latest_failed_at ?? 'n/a'}</strong></div>
          </div>
        </article>

        <article className="panel stack">
          <div>
            <h2>Next actions</h2>
            <p className="muted">Most common workflows are routed through the shared shell.</p>
          </div>
          <div className="quick-links">
            <Link to="/runs" className="quick-link-card"><strong>Runs</strong><span>Start, monitor, and inspect websocket-driven jobs.</span></Link>
            <Link to="/datasets" className="quick-link-card"><strong>Datasets</strong><span>Filter, preview, and launch runs with a selected dataset.</span></Link>
            <Link to="/profile" className="quick-link-card"><strong>Profile</strong><span>Review identity and account metadata.</span></Link>
            <Link to="/security" className="quick-link-card"><strong>Security</strong><span>Update your password and inspect access status.</span></Link>
          </div>
        </article>
      </section>
    </div>
  );
}
