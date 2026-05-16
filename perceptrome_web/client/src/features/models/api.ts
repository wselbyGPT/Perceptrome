import { apiRequest } from "../../lib/api-client";

export type ModelArtifact = {
  id: number;
  role: string;
  path: string;
  label?: string | null;
  download_url: string;
  created_at: string;
};

export type ModelVersion = {
  id: string;
  model_id: string;
  source_run_id?: string | null;
  version_label: string;
  status: string;
  architecture?: string | null;
  tokenizer?: string | null;
  checkpoint_path?: string | null;
  config_snapshot_path?: string | null;
  manifest_path?: string | null;
  metrics: Record<string, unknown>;
  metadata: Record<string, unknown>;
  created_at: string;
  promoted_at?: string | null;
  artifacts: ModelArtifact[];
};

export type RegisteredModel = {
  id: string;
  owner_user_id: string;
  name: string;
  description?: string | null;
  visibility: string;
  status: string;
  tags: string[];
  current_version_id?: string | null;
  created_at: string;
  updated_at: string;
  versions: ModelVersion[];
  current_version?: ModelVersion | null;
};

export type ModelRegistrySummary = {
  total_models: number;
  total_versions: number;
  architecture_counts: Record<string, number>;
  tokenizer_counts: Record<string, number>;
};

export type RegisterModelFromRunPayload = {
  run_id: string;
  model_id?: string | null;
  name?: string | null;
  description?: string | null;
  visibility?: string;
  tags?: string[];
  version_label?: string | null;
  version_status?: string;
};

export const modelsApi = {
  list: (filters?: { search?: string; architecture?: string; status?: string; visibility?: string; limit?: number }) => {
    const params = new URLSearchParams();
    if (filters?.search) params.set("search", filters.search);
    if (filters?.architecture) params.set("architecture", filters.architecture);
    if (filters?.status) params.set("status", filters.status);
    if (filters?.visibility) params.set("visibility", filters.visibility);
    params.set("limit", String(filters?.limit ?? 100));
    return apiRequest<RegisteredModel[]>(`/api/models?${params.toString()}`);
  },
  summary: () => apiRequest<ModelRegistrySummary>("/api/models/summary"),
  detail: (modelId: string) => apiRequest<RegisteredModel>(`/api/models/${encodeURIComponent(modelId)}`),
  registerFromRun: (payload: RegisterModelFromRunPayload) => (
    apiRequest<RegisteredModel>("/api/models/register-from-run", { method: "POST", body: payload })
  ),
  update: (modelId: string, payload: Partial<Pick<RegisteredModel, "name" | "description" | "visibility" | "status" | "tags" | "current_version_id">>) => (
    apiRequest<RegisteredModel>(`/api/models/${encodeURIComponent(modelId)}`, { method: "PATCH", body: payload })
  ),
  updateVersion: (modelId: string, versionId: string, payload: { version_label?: string; status?: string; promote_current?: boolean }) => (
    apiRequest<RegisteredModel>(`/api/models/${encodeURIComponent(modelId)}/versions/${encodeURIComponent(versionId)}`, { method: "PATCH", body: payload })
  ),
};
