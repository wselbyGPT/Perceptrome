export type RunArtifact = {
  id: number;
  phase?: string | null;
  path: string;
  label?: string | null;
  download_url: string;
  created_at: string;
};

export type RunRecord = {
  run_id: string;
  user_id: string;
  kind: string;
  state: string;
  message?: string | null;
  config: Record<string, unknown>;
  result?: Record<string, unknown> | null;
  submitted_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  artifacts: RunArtifact[];
};

export type RunSummary = {
  total_runs: number;
  state_counts: Record<string, number>;
  queued: number;
  running: number;
  completed: number;
  failed: number;
  canceled: number;
  latest_failed_run_id?: string | null;
  latest_failed_at?: string | null;
};

export type ActiveRunsResponse = {
  generated_at: string;
  runs: RunRecord[];
};

export type FailureRunsResponse = {
  generated_at: string;
  runs: RunRecord[];
};

export type ConfigSnapshot = {
  path: string;
  sha256: string;
  format?: string;
};

export type LineageNode = {
  id: string;
  kind: string;
  label: string;
  depth: number;
  run_id?: string | null;
  artifact_id?: string | null;
  artifact_type?: string | null;
  run_state?: string | null;
  path?: string | null;
  relation?: string | null;
  hash?: string | null;
  config_snapshot?: ConfigSnapshot | null;
  payload: Record<string, unknown>;
};

export type LineageEdge = {
  source: string;
  target: string;
  relation: string;
};

export type RunLineage = {
  run_id: string;
  depth_limit: number;
  artifact_type_filter?: string | null;
  run_state_filter: string[];
  nodes: LineageNode[];
  edges: LineageEdge[];
};

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, {
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
    ...init,
  });

  if (!resp.ok) {
    let detail = `${resp.status} ${resp.statusText}`;
    try {
      const body = await resp.json();
      detail = body?.detail ?? body?.message ?? detail;
    } catch {
      // noop
    }
    throw new Error(String(detail));
  }
  return (await resp.json()) as T;
}

export async function listRuns(limit = 50): Promise<RunRecord[]> {
  return fetchJson<RunRecord[]>(`/api/runs?limit=${limit}`);
}

export async function getRunsSummary(): Promise<RunSummary> {
  return fetchJson<RunSummary>("/api/runs/summary");
}

export async function listActiveRuns(limit = 12): Promise<ActiveRunsResponse> {
  return fetchJson<ActiveRunsResponse>(`/api/runs/active?limit=${limit}`);
}

export async function listFailedRuns(limit = 12): Promise<FailureRunsResponse> {
  return fetchJson<FailureRunsResponse>(`/api/runs/failures?limit=${limit}`);
}

export async function getRun(runId: string): Promise<RunRecord> {
  return fetchJson<RunRecord>(`/api/runs/${encodeURIComponent(runId)}`);
}

export async function getRunLineage(
  runId: string,
  options?: { depth?: number; artifactType?: string; runStates?: string[] },
): Promise<RunLineage> {
  const params = new URLSearchParams();
  if (typeof options?.depth === "number") params.set("depth", String(options.depth));
  if (options?.artifactType) params.set("artifact_type", options.artifactType);
  if (options?.runStates?.length) params.set("run_state", options.runStates.join(","));
  const suffix = params.toString() ? `?${params.toString()}` : "";
  return fetchJson<RunLineage>(`/api/runs/${encodeURIComponent(runId)}/lineage${suffix}`);
}
