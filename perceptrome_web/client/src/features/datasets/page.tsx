import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link, useNavigate } from "react-router-dom";
import { listDatasets, getDatasetPreview } from "../../dataset_api";
import { AppShell } from "../../app/app-shell";
import { useAuth } from "../auth/auth-context";

export function DatasetsPage() {
  const navigate = useNavigate();
  const { data = [], isLoading, error } = useQuery({ queryKey: ["datasets"], queryFn: listDatasets });
  const { me } = useAuth();
  const [search, setSearch] = useState("");
  const [source, setSource] = useState("");
  const [preview, setPreview] = useState("Select a dataset to view preview rows.");
  const sources = useMemo(() => Array.from(new Set(data.map((dataset) => dataset.source))).sort(), [data]);
  const filtered = useMemo(() => data.filter((dataset) => { if (source && dataset.source !== source) return false; const needle = search.trim().toLowerCase(); if (!needle) return true; return [dataset.dataset_id, dataset.source, ...(dataset.tags || [])].join(" ").toLowerCase().includes(needle); }), [data, search, source]);

  return (
    <div className="page-admin"><AppShell title="Dataset Catalog" subtitle="Search and filter datasets from config manifests." actions={me?.role === 'admin' ? <Link to="/admin/users" className="btn btn--secondary">Admin users</Link> : undefined}><div className="admin-wrap"><div className="panel"><div className="row"><label className="input-group"><span className="label">Search</span><input className="input" type="text" placeholder="dataset id, source, tag" value={search} onChange={(event) => setSearch(event.target.value)} /></label><label className="input-group"><span className="label">Source</span><select className="input" value={source} onChange={(event) => setSource(event.target.value)}><option value="">All sources</option>{sources.map((item) => <option key={item} value={item}>{item}</option>)}</select></label></div><div className={`msg${error ? ' error' : isLoading ? '' : ' ok'}`}>{error instanceof Error ? error.message : isLoading ? 'Loading datasets…' : `Showing ${filtered.length} / ${data.length} dataset(s).`}</div></div><div className="panel"><h2>Datasets</h2><div className="stack">{filtered.map((dataset) => { const splits = dataset.split_metadata.map((split) => `${split.name}:${split.count}`).join(' · ') || 'n/a'; return <div key={dataset.dataset_id} className="panel-body"><div className="toolbar" style={{ justifyContent: 'space-between', alignItems: 'start' }}><div className="stack"><strong>{dataset.dataset_id}</strong><div className="muted">source={dataset.source} | sequences={dataset.sequence_count}</div><div className="mono muted">hash={dataset.last_updated_hash.slice(0, 12)}…</div><div className="muted">splits: {splits}</div><div className="muted">tags: {(dataset.tags || []).join(', ') || 'n/a'}</div></div><div className="cluster"><button className="btn btn--secondary btn--sm" type="button" onClick={async () => { setPreview('Loading preview…'); try { const response = await getDatasetPreview(dataset.dataset_id, 25); setPreview(`${response.dataset_id} (${response.total_rows} rows)\n\n${response.preview.join('\n')}`); } catch (previewError) { setPreview(`Preview error: ${String(previewError)}`); } }}>Preview</button><button className="btn btn--primary btn--sm" type="button" onClick={() => navigate(`/runs?dataset=${encodeURIComponent(dataset.dataset_id)}&kind=stream&config_path=${encodeURIComponent('config/stream_config.yaml')}`)}>Use in run</button></div></div></div>; })}{!isLoading && !filtered.length ? <div className="muted">No datasets match the current filter.</div> : null}</div></div><div className="panel"><h2>Preview</h2><pre className="results-wrap">{preview}</pre></div></div></AppShell></div>
  );
}
