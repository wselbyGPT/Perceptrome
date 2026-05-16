import type { RegisteredModel, RegisterModelFromRunPayload } from "../models/api";
import type { RunRecord } from "./api";

const TRAINING_RUN_KINDS = new Set(["train_one", "stream", "pretrain"]);

function configParams(run: RunRecord): Record<string, unknown> {
  const params = run.config.params;
  if (params && typeof params === "object" && !Array.isArray(params)) {
    return { ...run.config, ...(params as Record<string, unknown>) };
  }
  return run.config;
}

function firstString(...values: unknown[]): string | null {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return null;
}

export function isRegisterableTrainingRun(run: RunRecord): boolean {
  return run.state === "completed" && TRAINING_RUN_KINDS.has(run.kind);
}

export function sourceRunModelMap(models: RegisteredModel[]): Map<string, RegisteredModel> {
  const out = new Map<string, RegisteredModel>();
  for (const model of models) {
    for (const version of model.versions) {
      if (version.source_run_id && !out.has(version.source_run_id)) {
        out.set(version.source_run_id, model);
      }
    }
  }
  return out;
}

export function defaultModelNameForRun(run: RunRecord): string {
  const params = configParams(run);
  const architecture = firstString(params.model_type, params.model_family, params.architecture);
  const tokenizer = firstString(params.tokenizer);
  const prefix = [architecture, tokenizer].filter(Boolean).join(" / ");
  return prefix ? `${prefix} ${run.kind} ${run.run_id}` : `${run.kind} ${run.run_id}`;
}

export function buildRegisterModelPayload(run: RunRecord): RegisterModelFromRunPayload {
  return {
    run_id: run.run_id,
    name: defaultModelNameForRun(run),
    visibility: "private",
    tags: [run.kind, "training"],
    version_status: "candidate",
  };
}
