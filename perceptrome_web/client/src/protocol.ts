// src/protocol.ts

export type RunConfig = {
  // Whatever knobs Perceptrome has:
  modelName: string;
  threshold: number;
  // etc.
};

export type ServerToClientMessage =
  | { type: "status"; status: "pending" | "running" | "done" | "error"; progress?: number }
  | { type: "log"; message: string }
  | { type: "result"; payload: any }; // tighten later

export type ClientToServerMessage =
  | { type: "start_run"; config: RunConfig };
