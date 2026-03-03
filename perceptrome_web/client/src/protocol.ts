// perceptrome_web/client/src/protocol.ts

/**
 * Perceptrome WebSocket protocol (client <-> server)
 *
 * Goals:
 * - Strong enough typing for current UI logic
 * - Flexible enough for protocol evolution (new fields/messages)
 * - Supports auth-aware status bootstrap, logs, results, and run control
 */

/* ---------------------------------------
 * Generic JSON types
 * ------------------------------------- */

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonObject | JsonValue[];
export type JsonObject = { [key: string]: JsonValue };

/* ---------------------------------------
 * Run configuration (client -> server)
 * ------------------------------------- */

/**
 * RunConfig is intentionally flexible for Perceptrome experiments.
 * Add stricter fields here as your backend schema stabilizes.
 */
export interface RunConfig {
  // Common identifiers
  run_id?: string;
  name?: string;
  job_name?: string;

  // Model / pipeline selection
  pipeline?: string;
  stage?: string;
  model?: string;
  model_family?: string;
  task?: string;

  // Data inputs
  dataset?: string;
  dataset_id?: string;
  input_path?: string;
  output_path?: string;

  // Training / execution parameters
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  seed?: number;
  device?: string; // e.g., "cpu", "cuda", "mps"
  max_steps?: number;
  timeout_seconds?: number;

  // Feature flags / options
  dry_run?: boolean;
  debug?: boolean;
  stream_logs?: boolean;
  save_artifacts?: boolean;

  // Nested parameter bags for future growth
  params?: Record<string, JsonValue>;
  tags?: string[];

  /**
   * Allow additional backend-specific fields without breaking frontend typing.
   */
  [key: string]: JsonValue | undefined;
}

/* ---------------------------------------
 * Client -> Server messages
 * ------------------------------------- */

export interface StartRunMessage {
  type: "start_run";
  config: RunConfig;
  request_id?: string;
}

export interface StopRunMessage {
  type: "stop_run";
  run_id?: string;
  request_id?: string;
}

export interface PingMessage {
  type: "ping";
  ts?: number;
  request_id?: string;
}

export interface SubscribeLogsMessage {
  type: "subscribe_logs";
  run_id?: string;
  tail?: number;
  request_id?: string;
}

export interface SubscribeResultsMessage {
  type: "subscribe_results";
  run_id?: string;
  request_id?: string;
}

export interface ClientHelloMessage {
  type: "client_hello";
  client?: string; // e.g. "perceptrome_web"
  version?: string;
  capabilities?: string[];
  request_id?: string;
}

export interface ClientCustomMessage {
  type: string;
  [key: string]: JsonValue | undefined;
}

/**
 * Union of known outbound messages.
 * Includes a permissive fallback so experimental messages don't block builds.
 */
export type ClientToServerMessage =
  | StartRunMessage
  | StopRunMessage
  | PingMessage
  | SubscribeLogsMessage
  | SubscribeResultsMessage
  | ClientHelloMessage
  | ClientCustomMessage;

/* ---------------------------------------
 * Server -> Client messages
 * ------------------------------------- */

export interface StatusMessage {
  type: "status";
  status: string;

  /**
   * Progress in [0,1] if present.
   * (Frontend also tolerates "percent" from legacy servers.)
   */
  progress?: number;
  percent?: number;

  // Optional run context
  run_id?: string;
  phase?: string;

  // Auth/bootstrap context (used by current backend example)
  user_id?: string;
  role?: string;

  // Correlation/debug
  request_id?: string;

  [key: string]: JsonValue | undefined;
}

export interface LogMessage {
  type: "log";
  line?: string;
  log?: string;
  message?: string;

  level?: "debug" | "info" | "warn" | "warning" | "error";
  ts?: string | number;
  run_id?: string;
  request_id?: string;

  [key: string]: JsonValue | undefined;
}

export interface LogsMessage {
  type: "logs";
  lines?: string[];
  logs?: string[];

  run_id?: string;
  request_id?: string;

  [key: string]: JsonValue | undefined;
}

export interface ResultMessage {
  type: "result";
  result?: JsonValue;
  data?: JsonValue;
  payload?: JsonValue;

  run_id?: string;
  request_id?: string;

  [key: string]: JsonValue | undefined;
}

export interface ResultsMessage {
  type: "results";
  results?: JsonValue;
  data?: JsonValue;
  payload?: JsonValue;

  run_id?: string;
  request_id?: string;

  [key: string]: JsonValue | undefined;
}

export interface ErrorMessage {
  type: "error";
  detail?: string;
  error?: string;
  message?: string;

  code?: string | number;
  run_id?: string;
  request_id?: string;

  [key: string]: JsonValue | undefined;
}

export interface RunStartedMessage {
  type: "run_started";
  run_id?: string;
  message?: string;
  [key: string]: JsonValue | undefined;
}

export interface RunStoppedMessage {
  type: "run_stopped";
  run_id?: string;
  message?: string;
  [key: string]: JsonValue | undefined;
}

export interface RunCompletedMessage {
  type: "run_completed";
  run_id?: string;
  result?: JsonValue;
  data?: JsonValue;
  message?: string;
  [key: string]: JsonValue | undefined;
}

export interface CompleteMessage {
  type: "complete";
  run_id?: string;
  result?: JsonValue;
  data?: JsonValue;
  message?: string;
  [key: string]: JsonValue | undefined;
}

export interface DoneMessage {
  type: "done";
  run_id?: string;
  result?: JsonValue;
  data?: JsonValue;
  message?: string;
  [key: string]: JsonValue | undefined;
}

export interface PongMessage {
  type: "pong";
  ts?: number;
  request_id?: string;
  [key: string]: JsonValue | undefined;
}

/**
 * Permissive fallback for future server messages.
 */
export interface ServerCustomMessage {
  type: string;
  [key: string]: JsonValue | undefined;
}

/**
 * Union of inbound messages understood by the current frontend.
 */
export type ServerToClientMessage =
  | StatusMessage
  | LogMessage
  | LogsMessage
  | ResultMessage
  | ResultsMessage
  | ErrorMessage
  | RunStartedMessage
  | RunStoppedMessage
  | RunCompletedMessage
  | CompleteMessage
  | DoneMessage
  | PongMessage
  | ServerCustomMessage;

/* ---------------------------------------
 * Runtime helpers (optional but useful)
 * ------------------------------------- */

export function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function hasTypeField(value: unknown): value is { type: string } {
  return isObject(value) && typeof value.type === "string";
}

export function isServerToClientMessage(value: unknown): value is ServerToClientMessage {
  return hasTypeField(value);
}

export function isClientToServerMessage(value: unknown): value is ClientToServerMessage {
  return hasTypeField(value);
}

export function parseServerMessage(raw: string): ServerToClientMessage {
  const parsed = JSON.parse(raw) as unknown;
  if (!isServerToClientMessage(parsed)) {
    throw new Error("Invalid server message: missing string 'type'");
  }
  return parsed;
}

export function serializeClientMessage(msg: ClientToServerMessage): string {
  return JSON.stringify(msg);
}

/* ---------------------------------------
 * Convenience builders (optional)
 * ------------------------------------- */

export function makeStartRunMessage(config: RunConfig, request_id?: string): StartRunMessage {
  return { type: "start_run", config, request_id };
}

export function makeStopRunMessage(run_id?: string, request_id?: string): StopRunMessage {
  return { type: "stop_run", run_id, request_id };
}

export function makePingMessage(request_id?: string): PingMessage {
  return { type: "ping", ts: Date.now(), request_id };
}
