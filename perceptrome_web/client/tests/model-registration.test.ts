import test from "node:test";
import assert from "node:assert/strict";

import { buildRegisterModelPayload, isRegisterableTrainingRun, sourceRunModelMap } from "../src/features/runs/model-registration";
import type { RegisteredModel } from "../src/features/models/api";
import type { RunRecord } from "../src/features/runs/api";

const completedStreamRun: RunRecord = {
  run_id: "run_stream_1",
  user_id: "user_1",
  kind: "stream",
  state: "completed",
  message: "done",
  config: { params: { model_type: "mamba", tokenizer: "base" } },
  result: null,
  submitted_at: "2026-05-12T00:00:00Z",
  artifacts: [],
};

test("completed training runs build a private model registration payload", () => {
  assert.equal(isRegisterableTrainingRun(completedStreamRun), true);

  const payload = buildRegisterModelPayload(completedStreamRun);

  assert.deepEqual(payload, {
    run_id: "run_stream_1",
    name: "mamba / base stream run_stream_1",
    visibility: "private",
    tags: ["stream", "training"],
    version_status: "candidate",
  });
});

test("source run model map marks already registered runs", () => {
  const models: RegisteredModel[] = [
    {
      id: "model_1",
      owner_user_id: "user_1",
      name: "Registered model",
      visibility: "private",
      status: "active",
      tags: [],
      created_at: "2026-05-12T00:00:00Z",
      updated_at: "2026-05-12T00:00:00Z",
      versions: [
        {
          id: "version_1",
          model_id: "model_1",
          source_run_id: "run_stream_1",
          version_label: "v1",
          status: "candidate",
          metrics: {},
          metadata: {},
          created_at: "2026-05-12T00:00:00Z",
          artifacts: [],
        },
      ],
    },
  ];

  assert.equal(sourceRunModelMap(models).get("run_stream_1")?.id, "model_1");
});
