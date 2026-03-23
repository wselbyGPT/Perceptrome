import { useEffect, useMemo, useRef, useState, type RefObject } from "react";
import { useSearchParams } from "react-router-dom";
import { setupPerceptromeViz } from "../../perceptrome_viz";
import { ActionFeedback, QueryBoundary } from "../../lib/query-helpers";
import { WorkspacePage } from "../../components/layout/workspace-page";
import { useDatasetCountQuery } from "../datasets/hooks";
import { useRunsLiveUpdates } from "./live-updates";
import { StatusBadge } from "../../components/ui/states";
import { BioAstExplorer } from "../bio-ast/explorer";
import { useRunsListQuery } from "./hooks";

function useVizMount(socket: WebSocket | null, rootRef: RefObject<HTMLDivElement | null>) {
  useEffect(() => {
    if (!socket || !rootRef.current) return;
    return setupPerceptromeViz(rootRef.current, socket);
  }, [socket, rootRef]);
}

export function RunsPage() {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const { socket, connectionState } = useRunsLiveUpdates();
  const datasetsQuery = useDatasetCountQuery();
  const runsQuery = useRunsListQuery(30);
  const [params] = useSearchParams();
  useVizMount(socket, rootRef);

  const selectedDataset = params.get('dataset') ?? '';
  const defaultBioAstRunId = params.get('bio_ast_run') ?? "";
  const defaultBioAstAccession = params.get('accession') ?? selectedDataset ?? "";
  const selectedKind = params.get('kind') ?? 'stream';
  const configPath = params.get('config_path') ?? 'config/stream_config.yaml';
  const runOptions = useMemo(() => {
    const ids = new Set((runsQuery.data ?? []).map((run) => run.run_id));
    if (defaultBioAstRunId) ids.add(defaultBioAstRunId);
    return Array.from(ids).sort();
  }, [runsQuery.data, defaultBioAstRunId]);
  const [bioAstRunId, setBioAstRunId] = useState<string>(defaultBioAstRunId);
  const [bioAstAccession, setBioAstAccession] = useState<string>(defaultBioAstAccession);

  const datasetOptions = useMemo(() => {
    const ids = new Set((datasetsQuery.data ?? []).map((dataset) => dataset.dataset_id));
    if (selectedDataset) ids.add(selectedDataset);
    return Array.from(ids).sort();
  }, [datasetsQuery.data, selectedDataset]);

  return (
    <WorkspacePage
      eyebrow="Runs workspace"
      title="Runs"
      description="Start experiments, watch websocket-backed status updates, and drill into lineage and artifacts from the authenticated shell."
      actions={<StatusBadge label={connectionState} tone={connectionState === 'connected' ? 'connected' : connectionState === 'connecting' ? 'pending' : 'disconnected'} />}
    >
      <QueryBoundary
        isLoading={datasetsQuery.isLoading || runsQuery.isLoading}
        error={datasetsQuery.error || runsQuery.error}
        loadingTitle="Loading runs workspace"
        loadingMessage="Preparing dataset options and live run controls."
        errorMessage="Failed to load run form data."
        onRetry={() => void datasetsQuery.refetch()}
      >
        <div className="stack-lg">
          <ActionFeedback title="Live runs visualization" message="The websocket-backed run console remains intact, but connection ownership now lives in a dedicated provider so the runs route can subscribe cleanly." tone="info" />
          <div className="panel"><div className="panel-header"><div><h2 className="panel-title">Bio-AST bundle loader</h2><p className="panel-subtitle">Incrementally mounted from the existing runs drill-down flow with additive bundle resolution.</p></div></div><div className="row"><label className="input-group"><span className="label">Run</span><select className="input" value={bioAstRunId} onChange={(event) => setBioAstRunId(event.target.value)}><option value="">Select a run</option>{runOptions.map((runId) => <option key={runId} value={runId}>{runId}</option>)}</select></label><label className="input-group"><span className="label">Accession</span><input className="input" value={bioAstAccession} onChange={(event) => setBioAstAccession(event.target.value)} placeholder="ACC123" /></label></div></div><main ref={rootRef} className="layout">
            <section className="column">
              <section className="panel" aria-labelledby="status-panel-title"><div className="panel-header"><div><h2 className="panel-title" id="status-panel-title">Run Queue / Status Board</h2><p className="panel-subtitle">Prioritized view of what is running and what failed recently</p></div></div><div className="status-card"><div className="status-row"><span className="status-dot" aria-hidden="true"></span><span className="status-label">Current status</span></div><div id="status" role="status" aria-live="polite">Waiting to connect…</div><div id="run-summary-cards" className="row"></div><h3 className="panel-title mt-2">Active runs</h3><div id="active-run-board" className="results-wrap"></div><h3 className="panel-title mt-2">Recent failures</h3><div id="failed-run-board" className="results-wrap"></div></div></section>
              <section className="panel"><div className="panel-header"><div><h2 className="panel-title">Start New Run</h2><p className="panel-subtitle">Existing start/stop controls preserved inside the shell experience.</p></div></div><form id="run-form" className="stack"><div className="row"><label className="input-group"><span className="label">Job kind</span><select id="run-kind" className="input" defaultValue={selectedKind}><option value="generate_plasmid">generate_plasmid</option><option value="generate_protein">generate_protein</option><option value="validate_plasmid">validate_plasmid</option><option value="pretrain">pretrain</option><option value="train_one">train_one</option><option value="stream">stream</option></select></label><label className="input-group"><span className="label">Config path</span><input id="config-path" className="input" defaultValue={configPath} /></label></div><div className="row"><label className="input-group"><span className="label">Dataset</span><select id="dataset" className="input" defaultValue={selectedDataset || datasetOptions[0] || ''}>{datasetOptions.map((datasetId) => <option key={datasetId} value={datasetId}>{datasetId}</option>)}</select></label><label className="input-group"><span className="label">Model family</span><select id="model-family" className="input" defaultValue="baseline"><option value="baseline">baseline</option><option value="vae">vae</option><option value="transformer">transformer</option></select></label></div><div className="row3"><label className="input-group"><span className="label">Temperature</span><input id="temperature" className="input" type="number" min="0.1" step="0.1" defaultValue="1.0" /></label><label className="input-group"><span className="label">Length bp</span><input id="length-bp" className="input" type="number" min="64" step="1" defaultValue="10000" /></label><button id="run-start-btn" className="btn btn--primary" type="submit">Start Run</button></div><div className="run-actions"><button id="run-stop-btn" className="btn btn--secondary" type="button">Stop Run</button><button id="refresh-history-btn" className="btn btn--secondary" type="button">Refresh History</button></div></form></section>
              <section id="run-drilldown" className="panel" aria-labelledby="lineage-panel-title"><div className="panel-header"><div><h2 className="panel-title" id="lineage-panel-title">Run Drill-down (Lineage + History)</h2><p className="panel-subtitle">Lineage, history, and artifact inspection remain intact.</p></div></div><h3 className="panel-title">Run history</h3><div id="run-history" className="results-wrap"></div><div className="row3 mt-2"><label className="input-group"><span className="label">Depth</span><input id="lineage-depth" className="input" type="number" min="0" max="6" defaultValue="2" /></label><label className="input-group"><span className="label">Artifact type</span><input id="lineage-artifact-type" className="input" placeholder="json, npy, ..." /></label><button id="refresh-lineage-btn" className="btn btn--secondary" type="button">Refresh Lineage</button></div><label className="input-group mt-2"><span className="label">Run states</span><select id="lineage-run-state" className="input" multiple><option value="queued">queued</option><option value="running">running</option><option value="completed">completed</option><option value="failed">failed</option><option value="canceled">canceled</option></select></label><div className="lineage-wrap mt-2"><svg id="lineage-graph" className="lineage-graph" viewBox="0 0 720 280" role="img" aria-label="Lineage graph"></svg></div><pre id="lineage-details" className="results-wrap mt-2">Select a run to view lineage.</pre></section>
            </section>
            <section className="column"><section className="panel" aria-labelledby="timeline-title"><div className="panel-header"><div><h2 className="panel-title" id="timeline-title">Active Run Timeline</h2><p className="panel-subtitle">Latest phases and events for the currently active run</p></div></div><div id="active-run-timeline" className="results-wrap"></div></section><section className="panel" aria-labelledby="logs-panel-title"><div className="panel-header"><div><h2 className="panel-title" id="logs-panel-title">Live Logs</h2><p className="panel-subtitle">Server messages and streaming job output</p></div><div className="run-actions"><button id="clear-logs-btn" className="btn btn--secondary btn--sm" type="button">Clear</button></div></div><div id="logs" className="terminal" aria-live="polite"></div></section><section className="panel"><div className="panel-header"><div><h2 className="panel-title">Metrics Panel</h2><p className="panel-subtitle">Streaming metrics</p></div></div><pre id="metrics" className="results-wrap"></pre></section><section className="panel"><div className="panel-header"><div><h2 className="panel-title">Artifact Summary</h2><p className="panel-subtitle">Current run outputs and downloadable artifacts</p></div></div><div id="results" className="results-wrap" aria-live="polite"></div><h3 className="panel-title mt-2">Generated Sequences</h3><pre id="generated-sequences" className="results-wrap"></pre><h3 className="panel-title mt-2">Validation</h3><pre id="validation-results" className="results-wrap"></pre><pre id="checkpoints" className="results-wrap"></pre></section></section>
          </main>
          <BioAstExplorer runId={bioAstRunId || null} accession={bioAstAccession || undefined} />
        </div>
      </QueryBoundary>
    </WorkspacePage>
  );
}
