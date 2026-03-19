export {
  runsApi,
  type ConfigSnapshot,
  type LineageEdge,
  type LineageNode,
  type RunArtifact,
  type RunLineage,
  type RunRecord,
  type RunSummary,
  type RunsBoardResponse as ActiveRunsResponse,
  type RunsBoardResponse as FailureRunsResponse,
} from "./features/runs/api";
import { runsApi } from "./features/runs/api";

export const listRuns = (limit = 50) => runsApi.list(limit);
export const getRunsSummary = () => runsApi.summary();
export const listActiveRuns = (limit = 12) => runsApi.active(limit);
export const listFailedRuns = (limit = 12) => runsApi.failures(limit);
export const getRun = (runId: string) => runsApi.detail(runId);
export const getRunLineage = (runId: string, options?: { depth?: number; artifactType?: string; runStates?: string[] }) => runsApi.lineage(runId, options);
