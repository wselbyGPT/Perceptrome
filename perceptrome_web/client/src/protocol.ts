export type ServerToClientMessage =
  | { type: "status"; status: "pending" | "running" | "done" | "error"; progress?: number | null }
  | { type: "log"; message: string }
  | { type: "result"; payload: { command: string; cwd: string; exit_code: number; ok: boolean } };

export type ClientToServerMessage =
  | { type: "start_run"; command: string; cwd?: string }
  | { type: "stop_run" };
