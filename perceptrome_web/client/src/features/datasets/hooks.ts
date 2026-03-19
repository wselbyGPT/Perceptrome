import { useQuery } from "@tanstack/react-query";
import { datasetsApi } from "./api";

export const datasetQueryKeys = {
  all: ["datasets"] as const,
  lists: () => [...datasetQueryKeys.all, "list"] as const,
  count: () => [...datasetQueryKeys.all, "count"] as const,
  detail: (datasetId: string) => [...datasetQueryKeys.all, "detail", datasetId] as const,
  preview: (datasetId: string, limit: number) => [...datasetQueryKeys.all, "preview", datasetId, limit] as const,
  runForm: () => [...datasetQueryKeys.all, "run-form"] as const,
};

export function useDatasetsQuery() {
  return useQuery({ queryKey: datasetQueryKeys.lists(), queryFn: datasetsApi.list });
}

export function useDatasetCountQuery() {
  return useQuery({ queryKey: datasetQueryKeys.count(), queryFn: datasetsApi.list });
}

export function useDatasetDetailQuery(datasetId: string | null) {
  return useQuery({ queryKey: datasetId ? datasetQueryKeys.detail(datasetId) : [...datasetQueryKeys.all, "detail", "none"], queryFn: () => datasetsApi.detail(datasetId!), enabled: Boolean(datasetId) });
}

export function useDatasetPreviewQuery(datasetId: string | null, limit = 25) {
  return useQuery({ queryKey: datasetId ? datasetQueryKeys.preview(datasetId, limit) : [...datasetQueryKeys.all, "preview", "none", limit], queryFn: () => datasetsApi.preview(datasetId!, limit), enabled: Boolean(datasetId) });
}
