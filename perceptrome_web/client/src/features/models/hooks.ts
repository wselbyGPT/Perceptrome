import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { modelsApi, type RegisterModelFromRunPayload } from "./api";

export const modelQueryKeys = {
  all: ["models"] as const,
  list: (filters: { search?: string; architecture?: string; status?: string; visibility?: string }) => [...modelQueryKeys.all, "list", filters] as const,
  summary: () => [...modelQueryKeys.all, "summary"] as const,
  detail: (modelId: string) => [...modelQueryKeys.all, "detail", modelId] as const,
};

export function useModelsQuery(filters: { search?: string; architecture?: string; status?: string; visibility?: string } = {}) {
  return useQuery({ queryKey: modelQueryKeys.list(filters), queryFn: () => modelsApi.list(filters) });
}

export function useModelSummaryQuery() {
  return useQuery({ queryKey: modelQueryKeys.summary(), queryFn: modelsApi.summary });
}

export function useModelDetailQuery(modelId: string | null) {
  return useQuery({
    queryKey: modelId ? modelQueryKeys.detail(modelId) : [...modelQueryKeys.all, "detail", "none"],
    queryFn: () => modelsApi.detail(modelId!),
    enabled: Boolean(modelId),
  });
}

export function useRegisterModelFromRunMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (payload: RegisterModelFromRunPayload) => modelsApi.registerFromRun(payload),
    onSuccess: (model) => {
      void queryClient.invalidateQueries({ queryKey: modelQueryKeys.all });
      queryClient.setQueryData(modelQueryKeys.detail(model.id), model);
    },
  });
}

export function useUpdateModelMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ modelId, payload }: { modelId: string; payload: Parameters<typeof modelsApi.update>[1] }) => modelsApi.update(modelId, payload),
    onSuccess: (model) => {
      void queryClient.invalidateQueries({ queryKey: modelQueryKeys.all });
      queryClient.setQueryData(modelQueryKeys.detail(model.id), model);
    },
  });
}

export function useUpdateModelVersionMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ modelId, versionId, payload }: { modelId: string; versionId: string; payload: Parameters<typeof modelsApi.updateVersion>[2] }) => modelsApi.updateVersion(modelId, versionId, payload),
    onSuccess: (model) => {
      void queryClient.invalidateQueries({ queryKey: modelQueryKeys.all });
      queryClient.setQueryData(modelQueryKeys.detail(model.id), model);
    },
  });
}
