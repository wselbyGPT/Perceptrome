import { useQuery } from "@tanstack/react-query";
import { runsApi } from "./api";

export const runQueryKeys = {
  all: ["runs"] as const,
  list: (limit = 50) => [...runQueryKeys.all, "list", limit] as const,
  summary: () => [...runQueryKeys.all, "summary"] as const,
  active: (limit = 12) => [...runQueryKeys.all, "active", limit] as const,
  failures: (limit = 12) => [...runQueryKeys.all, "failures", limit] as const,
  detail: (runId: string) => [...runQueryKeys.all, "detail", runId] as const,
  lineage: (runId: string, depth: number, artifactType: string, runStates: string[]) => [...runQueryKeys.all, "lineage", runId, depth, artifactType, runStates.join(",")] as const,
};

export function useRunSummaryQuery() {
  return useQuery({ queryKey: runQueryKeys.summary(), queryFn: runsApi.summary });
}

export function useRunsListQuery(limit = 50) {
  return useQuery({ queryKey: runQueryKeys.list(limit), queryFn: () => runsApi.list(limit) });
}

export function useActiveRunsQuery(limit = 12) {
  return useQuery({ queryKey: runQueryKeys.active(limit), queryFn: () => runsApi.active(limit) });
}

export function useFailedRunsQuery(limit = 12) {
  return useQuery({ queryKey: runQueryKeys.failures(limit), queryFn: () => runsApi.failures(limit) });
}

export function useRunDetailQuery(runId: string | null) {
  return useQuery({ queryKey: runId ? runQueryKeys.detail(runId) : [...runQueryKeys.all, "detail", "none"], queryFn: () => runsApi.detail(runId!), enabled: Boolean(runId) });
}
