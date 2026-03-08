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

export async function getRun(runId: string): Promise<RunRecord> {
  return fetchJson<RunRecord>(`/api/runs/${encodeURIComponent(runId)}`);
}
