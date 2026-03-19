import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { EmptyState, StatusBadge } from "../../components/ui/states";
import { ActionFeedback, QueryBoundary } from "../../lib/query-helpers";
import { WorkspacePage } from "../../components/layout/workspace-page";
import { useDatasetDetailQuery, useDatasetPreviewQuery, useDatasetsQuery } from "./hooks";
import type { DatasetCatalogItem } from "./api";

function datasetMatches(dataset: DatasetCatalogItem, search: string, source: string, tag: string) {
  if (source && dataset.source !== source) return false;
  if (tag && !(dataset.tags || []).includes(tag)) return false;
  const needle = search.trim().toLowerCase();
  if (!needle) return true;
  return [dataset.dataset_id, dataset.source, ...(dataset.tags || [])].join(" ").toLowerCase().includes(needle);
}

export function DatasetsPage() {
  const navigate = useNavigate();
  const datasetsQuery = useDatasetsQuery();
  const [search, setSearch] = useState("");
  const [source, setSource] = useState("");
  const [tag, setTag] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const previewQuery = useDatasetPreviewQuery(selectedId, 25);
  const detailQuery = useDatasetDetailQuery(selectedId);

  const data = datasetsQuery.data ?? [];
  const sources = useMemo(() => Array.from(new Set(data.map((dataset) => dataset.source))).sort(), [data]);
  const tags = useMemo(() => Array.from(new Set(data.flatMap((dataset) => dataset.tags || []))).sort(), [data]);
  const filtered = useMemo(() => data.filter((dataset) => datasetMatches(dataset, search, source, tag)), [data, search, source, tag]);
  const selectedDataset = useMemo(() => filtered.find((dataset) => dataset.dataset_id === selectedId) ?? data.find((dataset) => dataset.dataset_id === selectedId) ?? filtered[0] ?? null, [data, filtered, selectedId]);

  useEffect(() => {
    if (!selectedId && filtered[0]) setSelectedId(filtered[0].dataset_id);
    if (selectedId && !filtered.some((dataset) => dataset.dataset_id === selectedId) && filtered[0]) {
      setSelectedId(filtered[0].dataset_id);
    }
  }, [filtered, selectedId]);

  return (
    <WorkspacePage
      eyebrow="Datasets workspace"
      title="Datasets"
      description="Browse the authenticated dataset catalog, inspect details and preview rows, and launch runs with shared loading, error, and feedback patterns."
      actions={<button className="btn btn--secondary" type="button" onClick={() => void datasetsQuery.refetch()}>Refresh catalog</button>}
    >
      <QueryBoundary
        isLoading={datasetsQuery.isLoading}
        error={datasetsQuery.error}
        loadingTitle="Loading datasets"
        loadingMessage="Building the routed dataset catalog and preview experience."
        errorMessage="Failed to load datasets."
        onRetry={() => void datasetsQuery.refetch()}
      >
        <div className="stack-lg">
          <section className="panel stack">
            <div className="panel-header"><div><h2 className="panel-title">Dataset catalog</h2><p className="panel-subtitle">Runs and datasets now share the same shell navigation, loading states, and action treatment.</p></div></div>
            <div className="filters-grid">
              <label className="input-group"><span className="label">Search</span><input className="input" type="text" placeholder="dataset id, source, tag" value={search} onChange={(event) => setSearch(event.target.value)} /></label>
              <label className="input-group"><span className="label">Source</span><select className="input" value={source} onChange={(event) => setSource(event.target.value)}><option value="">All sources</option>{sources.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
              <label className="input-group"><span className="label">Tag</span><select className="input" value={tag} onChange={(event) => setTag(event.target.value)}><option value="">All tags</option>{tags.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
            </div>
            <ActionFeedback title="Catalog summary" message={`Showing ${filtered.length} of ${data.length} dataset(s). Select a row to inspect detail and preview content, then launch a run.`} tone="info" />
          </section>

          <section className="content-grid content-grid--datasets">
            <article className="panel stack">
              <div className="cluster" style={{ justifyContent: 'space-between' }}>
                <h2>Available datasets</h2>
                <StatusBadge label={`${filtered.length} visible`} tone="neutral" />
              </div>
              {!filtered.length ? (
                <EmptyState title="No matching datasets" message="Adjust the search, source, or tag filters to find datasets." />
              ) : (
                <div className="dataset-list">
                  {filtered.map((dataset) => {
                    const isSelected = dataset.dataset_id === selectedDataset?.dataset_id;
                    const splits = dataset.split_metadata.map((split) => `${split.name}:${split.count}`).join(' · ') || 'n/a';
                    return (
                      <button key={dataset.dataset_id} type="button" className={`dataset-list__item${isSelected ? ' dataset-list__item--selected' : ''}`} onClick={() => setSelectedId(dataset.dataset_id)}>
                        <div className="stack-sm">
                          <div className="cluster" style={{ justifyContent: 'space-between' }}><strong>{dataset.dataset_id}</strong><StatusBadge label={dataset.source} tone="info" /></div>
                          <span className="muted">Sequences: {dataset.sequence_count}</span>
                          <span className="muted">Splits: {splits}</span>
                          <div className="cluster">{(dataset.tags || []).map((item) => <StatusBadge key={item} label={item} tone="neutral" />)}</div>
                        </div>
                      </button>
                    );
                  })}
                </div>
              )}
            </article>

            <article className="panel stack">
              {selectedDataset ? (
                <>
                  <div className="stack-sm">
                    <div className="cluster" style={{ justifyContent: 'space-between' }}>
                      <div>
                        <h2>{selectedDataset.dataset_id}</h2>
                        <p className="muted">Source: {selectedDataset.source}</p>
                      </div>
                      <button className="btn btn--primary" type="button" onClick={() => navigate(`/runs?dataset=${encodeURIComponent(selectedDataset.dataset_id)}&kind=stream&config_path=${encodeURIComponent('config/stream_config.yaml')}`)}>Use in run</button>
                    </div>
                    <div className="definition-grid">
                      <div><span className="label">Sequence count</span><strong>{selectedDataset.sequence_count}</strong></div>
                      <div><span className="label">Manifest hash</span><strong className="mono">{selectedDataset.last_updated_hash}</strong></div>
                      <div><span className="label">Manifest path</span><strong className="mono">{detailQuery.data?.manifest_path ?? 'Loading…'}</strong></div>
                    </div>
                  </div>
                  <ActionFeedback title="Preview panel" message="Detail and preview data load through typed TanStack Query hooks only for the selected dataset to keep the workspace responsive." tone="success" />
                  <QueryBoundary
                    isLoading={previewQuery.isLoading || detailQuery.isLoading}
                    error={previewQuery.error ?? detailQuery.error}
                    loadingTitle="Loading dataset detail"
                    loadingMessage="Fetching manifest metadata and preview rows for the selected dataset."
                    errorMessage="Failed to load dataset detail."
                    onRetry={() => {
                      void previewQuery.refetch();
                      void detailQuery.refetch();
                    }}
                  >
                    <pre className="results-wrap dataset-preview">{previewQuery.data ? `${previewQuery.data.dataset_id} (${previewQuery.data.total_rows} rows)\n\n${previewQuery.data.preview.join('\n')}` : 'Select a dataset to view preview rows.'}</pre>
                  </QueryBoundary>
                </>
              ) : (
                <EmptyState title="Select a dataset" message="Choose a dataset from the list to inspect preview data and launch a run." />
              )}
            </article>
          </section>
        </div>
      </QueryBoundary>
    </WorkspacePage>
  );
}
