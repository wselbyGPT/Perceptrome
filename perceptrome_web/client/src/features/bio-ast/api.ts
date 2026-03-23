import { apiRequest } from "../../lib/api-client";
import { bioAstVisualizationBundleSchema, type BioASTVisualizationBundle } from "./schemas";

export async function getBioAstVisualizationBundle(runId: string, options?: { accession?: string; artifactId?: number }) {
  const params = new URLSearchParams();
  if (options?.accession) params.set("accession", options.accession);
  if (typeof options?.artifactId === "number") params.set("artifact_id", String(options.artifactId));
  const suffix = params.toString() ? `?${params.toString()}` : "";
  const payload = await apiRequest<unknown>(`/api/runs/${encodeURIComponent(runId)}/bio-ast${suffix}`);
  return bioAstVisualizationBundleSchema.parse(payload) as BioASTVisualizationBundle;
}
