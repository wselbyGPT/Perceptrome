import { useEffect, useMemo, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { WorkspacePage } from "../../components/layout/workspace-page";
import { EmptyState, StatusBadge } from "../../components/ui/states";
import { ActionFeedback, QueryBoundary } from "../../lib/query-helpers";
import { useRunsListQuery } from "../runs/hooks";
import type { RegisteredModel, RegisterModelFromRunPayload } from "./api";
import {
  useModelSummaryQuery,
  useModelsQuery,
  useRegisterModelFromRunMutation,
  useUpdateModelMutation,
  useUpdateModelVersionMutation,
} from "./hooks";

function parseTags(value: string): string[] {
  return value.split(",").map((tag) => tag.trim()).filter(Boolean);
}

function formatDate(value?: string | null): string {
  if (!value) return "n/a";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function compactPath(value?: string | null): string {
  if (!value) return "n/a";
  const parts = value.split("/");
  return parts.length > 3 ? `${parts.slice(0, 2).join("/")}/.../${parts.slice(-2).join("/")}` : value;
}

function ModelList({
  models,
  selectedId,
  onSelect,
}: {
  models: RegisteredModel[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  if (!models.length) {
    return <EmptyState title="No registered models" message="Register a completed run to create the first model entry." />;
  }

  return (
    <div className="dataset-list">
      {models.map((model) => {
        const current = model.current_version;
        return (
          <button
            key={model.id}
            type="button"
            className={`dataset-list__item${model.id === selectedId ? " dataset-list__item--selected" : ""}`}
            onClick={() => onSelect(model.id)}
          >
            <div className="cluster">
              <strong>{model.name}</strong>
              <StatusBadge label={model.status} />
              <StatusBadge label={model.visibility} tone="neutral" />
            </div>
            <span className="muted">{model.description || "No description"}</span>
            <span className="muted">
              {current?.architecture ?? "unknown architecture"} · {current?.tokenizer ?? "unknown tokenizer"} · {model.versions.length} version(s)
            </span>
            <div className="cluster">
              {model.tags.map((tag) => <StatusBadge key={tag} label={tag} tone="neutral" />)}
            </div>
          </button>
        );
      })}
    </div>
  );
}

function RegisterRunForm({ models, onRegistered }: { models: RegisteredModel[]; onRegistered: (modelId: string) => void }) {
  const runsQuery = useRunsListQuery(100);
  const registerMutation = useRegisterModelFromRunMutation();
  const completedRuns = useMemo(() => (runsQuery.data ?? []).filter((run) => run.state === "completed"), [runsQuery.data]);
  const [targetModelId, setTargetModelId] = useState("");
  const [runId, setRunId] = useState("");
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [tags, setTags] = useState("");
  const [visibility, setVisibility] = useState("private");
  const [versionLabel, setVersionLabel] = useState("");
  const [versionStatus, setVersionStatus] = useState("candidate");
  const targetModel = useMemo(() => models.find((model) => model.id === targetModelId) ?? null, [models, targetModelId]);

  useEffect(() => {
    setVisibility(targetModel?.visibility ?? "private");
  }, [targetModel]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!runId) return;
    const payload: RegisterModelFromRunPayload = {
      run_id: runId,
      model_id: targetModelId || undefined,
      version_label: versionLabel || undefined,
      version_status: versionStatus,
    };
    if (!targetModelId || name.trim()) payload.name = name || undefined;
    if (!targetModelId || description.trim()) payload.description = description || undefined;
    if (!targetModelId || tags.trim()) payload.tags = parseTags(tags);
    if (!targetModelId || visibility !== targetModel?.visibility) payload.visibility = visibility;
    const model = await registerMutation.mutateAsync(payload);
    onRegistered(model.id);
    setRunId("");
    setVersionLabel("");
    if (!targetModelId) {
      setName("");
      setDescription("");
      setTags("");
    }
  };

  return (
    <form className="stack" onSubmit={(event) => void submit(event)}>
      <div className="row">
        <label className="input-group"><span className="label">Completed run</span>
          <select className="input" value={runId} onChange={(event) => setRunId(event.target.value)} required>
            <option value="">Select a completed run</option>
            {completedRuns.map((run) => <option key={run.run_id} value={run.run_id}>{run.run_id} · {run.kind}</option>)}
          </select>
        </label>
        <label className="input-group"><span className="label">Target model</span>
          <select className="input" value={targetModelId} onChange={(event) => setTargetModelId(event.target.value)}>
            <option value="">Create new model</option>
            {models.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
          </select>
        </label>
      </div>
      <div className="row">
        <label className="input-group"><span className="label">Model name</span>
          <input className="input" value={name} onChange={(event) => setName(event.target.value)} placeholder={targetModelId ? "leave unchanged" : "e.g. Viral DNA Mamba v1"} />
        </label>
        <label className="input-group"><span className="label">Version label</span>
          <input className="input" value={versionLabel} onChange={(event) => setVersionLabel(event.target.value)} placeholder="auto" />
        </label>
      </div>
      <label className="input-group"><span className="label">Description</span>
        <textarea className="input" value={description} onChange={(event) => setDescription(event.target.value)} rows={3} placeholder="Purpose, source data, or intended use." />
      </label>
      <div className="row">
        <label className="input-group"><span className="label">Tags</span>
          <input className="input" value={tags} onChange={(event) => setTags(event.target.value)} placeholder="dna, mamba, plasmid" />
        </label>
        <label className="input-group"><span className="label">Visibility</span>
          <select className="input" value={visibility} onChange={(event) => setVisibility(event.target.value)}>
            <option value="private">private</option>
            <option value="team">team</option>
            <option value="public">public</option>
          </select>
        </label>
      </div>
      <div className="row">
        <label className="input-group"><span className="label">Version status</span>
          <select className="input" value={versionStatus} onChange={(event) => setVersionStatus(event.target.value)}>
            <option value="candidate">candidate</option>
            <option value="stable">stable</option>
            <option value="deprecated">deprecated</option>
            <option value="archived">archived</option>
          </select>
        </label>
        <div className="input-group">
          <span className="label">Action</span>
          <button className="btn btn--primary" type="submit" disabled={!runId || registerMutation.isPending}>
            {targetModelId ? "Add Version" : "Register Model"}
          </button>
        </div>
      </div>
      {registerMutation.error ? <ActionFeedback title="Registration failed" message={registerMutation.error instanceof Error ? registerMutation.error.message : "Could not register model."} tone="error" /> : null}
      {registerMutation.isSuccess ? <ActionFeedback title="Registered" message="The model registry entry is up to date." tone="success" /> : null}
    </form>
  );
}

function ModelDetail({ model }: { model: RegisteredModel }) {
  const updateModel = useUpdateModelMutation();
  const updateVersion = useUpdateModelVersionMutation();
  const [tags, setTags] = useState(model.tags.join(", "));

  useEffect(() => {
    setTags(model.tags.join(", "));
  }, [model.id, model.tags]);

  const current = model.current_version;
  const saveTags = () => {
    updateModel.mutate({ modelId: model.id, payload: { tags: parseTags(tags) } });
  };
  const setModelStatus = (status: string) => {
    updateModel.mutate({ modelId: model.id, payload: { status } });
  };
  const promoteVersion = (versionId: string) => {
    updateVersion.mutate({ modelId: model.id, versionId, payload: { promote_current: true } });
  };
  const archiveVersion = (versionId: string) => {
    updateVersion.mutate({ modelId: model.id, versionId, payload: { status: "archived" } });
  };

  return (
    <div className="stack-lg">
      <section className="panel">
        <div className="panel-header">
          <div>
            <h2 className="panel-title">{model.name}</h2>
            <p className="panel-subtitle">{model.description || "No description"}</p>
          </div>
          <div className="cluster">
            <StatusBadge label={model.status} />
            <StatusBadge label={model.visibility} tone="neutral" />
          </div>
        </div>
        <div className="definition-grid">
          <div className="kv-row"><span>Current architecture</span><strong>{current?.architecture ?? "n/a"}</strong></div>
          <div className="kv-row"><span>Tokenizer</span><strong>{current?.tokenizer ?? "n/a"}</strong></div>
          <div className="kv-row"><span>Versions</span><strong>{model.versions.length}</strong></div>
          <div className="kv-row"><span>Updated</span><strong>{formatDate(model.updated_at)}</strong></div>
        </div>
        <div className="row mt-2">
          <label className="input-group"><span className="label">Tags</span>
            <input className="input" value={tags} onChange={(event) => setTags(event.target.value)} />
          </label>
          <div className="input-group">
            <span className="label">Model actions</span>
            <div className="cluster">
              <button className="btn btn--secondary" type="button" onClick={saveTags}>Save Tags</button>
              <button className="btn btn--secondary" type="button" onClick={() => setModelStatus(model.status === "archived" ? "active" : "archived")}>
                {model.status === "archived" ? "Restore" : "Archive"}
              </button>
              {current?.source_run_id ? <Link className="btn btn--secondary" to={`/runs?bio_ast_run=${encodeURIComponent(current.source_run_id)}`}>Source Run</Link> : null}
            </div>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <h2 className="panel-title">Versions</h2>
            <p className="panel-subtitle">Immutable snapshots registered from completed runs.</p>
          </div>
        </div>
        <div className="stack">
          {model.versions.map((version) => (
            <article key={version.id} className="quick-link-card quick-link-card--static">
              <div className="cluster">
                <strong>{version.version_label}</strong>
                <StatusBadge label={version.status} />
                {model.current_version_id === version.id ? <StatusBadge label="current" tone="success" /> : null}
              </div>
              <div className="definition-grid">
                <div className="kv-row"><span>Architecture</span><strong>{version.architecture ?? "n/a"}</strong></div>
                <div className="kv-row"><span>Tokenizer</span><strong>{version.tokenizer ?? "n/a"}</strong></div>
                <div className="kv-row"><span>Checkpoint</span><strong className="mono">{compactPath(version.checkpoint_path)}</strong></div>
                <div className="kv-row"><span>Created</span><strong>{formatDate(version.created_at)}</strong></div>
              </div>
              <div className="cluster">
                <button className="btn btn--secondary btn--sm" type="button" onClick={() => promoteVersion(version.id)}>Promote</button>
                <button className="btn btn--secondary btn--sm" type="button" onClick={() => archiveVersion(version.id)}>Archive Version</button>
                {version.source_run_id ? <Link className="btn btn--secondary btn--sm" to={`/runs?bio_ast_run=${encodeURIComponent(version.source_run_id)}`}>Open Run</Link> : null}
              </div>
              <div className="table-wrap">
                <table className="table">
                  <thead><tr><th>Artifact</th><th>Role</th><th>Path</th><th></th></tr></thead>
                  <tbody>
                    {version.artifacts.map((artifact) => (
                      <tr key={artifact.id}>
                        <td>{artifact.label ?? artifact.role}</td>
                        <td>{artifact.role}</td>
                        <td className="mono">{compactPath(artifact.path)}</td>
                        <td><a className="btn btn--secondary btn--sm" href={artifact.download_url}>Download</a></td>
                      </tr>
                    ))}
                    {!version.artifacts.length ? <tr><td colSpan={4}>No downloadable artifacts were found for this version.</td></tr> : null}
                  </tbody>
                </table>
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

export function ModelsPage() {
  const [search, setSearch] = useState("");
  const [architecture, setArchitecture] = useState("");
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const filters = useMemo(() => ({ search, architecture }), [search, architecture]);
  const modelsQuery = useModelsQuery(filters);
  const summaryQuery = useModelSummaryQuery();
  const models = modelsQuery.data ?? [];
  const selectedModel = models.find((model) => model.id === selectedModelId) ?? models[0] ?? null;

  useEffect(() => {
    if (selectedModel && selectedModel.id !== selectedModelId) {
      setSelectedModelId(selectedModel.id);
    }
  }, [selectedModelId, selectedModel]);

  const architectures = Object.keys(summaryQuery.data?.architecture_counts ?? {}).sort();

  return (
    <WorkspacePage
      eyebrow="Model registry"
      title="Models"
      description="Register completed runs as versioned models, inspect lineage artifacts, and promote stable checkpoints."
      actions={<StatusBadge label={`${summaryQuery.data?.total_models ?? 0} models`} tone="neutral" />}
    >
      <QueryBoundary
        isLoading={modelsQuery.isLoading || summaryQuery.isLoading}
        error={modelsQuery.error || summaryQuery.error}
        loadingTitle="Loading model registry"
        loadingMessage="Preparing registered models and completed run options."
        errorMessage="Failed to load model registry."
        onRetry={() => { void modelsQuery.refetch(); void summaryQuery.refetch(); }}
      >
        <div className="stats-grid">
          <article className="panel stat-panel"><span className="label">Models</span><strong>{summaryQuery.data?.total_models ?? 0}</strong><StatusBadge label="registry" tone="neutral" /></article>
          <article className="panel stat-panel"><span className="label">Versions</span><strong>{summaryQuery.data?.total_versions ?? 0}</strong><StatusBadge label="versioned" tone="info" /></article>
          <article className="panel stat-panel"><span className="label">Architectures</span><strong>{architectures.length}</strong><StatusBadge label="diverse" tone="success" /></article>
          <article className="panel stat-panel"><span className="label">Current filter</span><strong>{models.length}</strong><StatusBadge label="visible" tone="neutral" /></article>
        </div>

        <section className="content-grid content-grid--datasets">
          <div className="stack-lg">
            <section className="panel">
              <div className="panel-header">
                <div>
                  <h2 className="panel-title">Register Run</h2>
                  <p className="panel-subtitle">Create a model or append a new immutable version from a completed run.</p>
                </div>
              </div>
              <RegisterRunForm models={models} onRegistered={setSelectedModelId} />
            </section>

            <section className="panel">
              <div className="panel-header">
                <div>
                  <h2 className="panel-title">Model Library</h2>
                  <p className="panel-subtitle">Search by name, description, tag, or architecture.</p>
                </div>
              </div>
              <div className="row">
                <label className="input-group"><span className="label">Search</span>
                  <input className="input" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="model name, tag, purpose" />
                </label>
                <label className="input-group"><span className="label">Architecture</span>
                  <select className="input" value={architecture} onChange={(event) => setArchitecture(event.target.value)}>
                    <option value="">all</option>
                    {architectures.map((item) => <option key={item} value={item}>{item}</option>)}
                  </select>
                </label>
              </div>
              <ModelList models={models} selectedId={selectedModelId} onSelect={setSelectedModelId} />
            </section>
          </div>

          <div>
            {selectedModel ? <ModelDetail model={selectedModel} /> : <EmptyState title="Select a model" message="Registered model details will appear here." />}
          </div>
        </section>
      </QueryBoundary>
    </WorkspacePage>
  );
}
