import type {
  RunConfig,
  ServerToClientMessage,
  ClientToServerMessage,
} from "./protocol";

type JsonRecord = Record<string, unknown>;

function mustEl<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing required element #${id}`);
  return el as T;
}

function asPrettyText(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function nowStamp(): string {
  const d = new Date();
  return d.toLocaleTimeString();
}

export function setupPerceptromeViz(ws: WebSocket) {
  const statusEl = mustEl<HTMLElement>("status");
  const logsEl = mustEl<HTMLElement>("logs");
  const resultsEl = mustEl<HTMLElement>("results");

  const startBtn = document.getElementById("run-start-btn") as HTMLButtonElement | null;
  const stopBtn = document.getElementById("run-stop-btn") as HTMLButtonElement | null;
  const clearLogsBtn = document.getElementById("clear-logs-btn") as HTMLButtonElement | null;

  let socketOpen = false;
  let runActive = false;
  let activeRunId: string | null = null;

  // -----------------------------
  // UI helpers
  // -----------------------------
  function setStatus(text: string, progress?: number | null) {
    if (typeof progress === "number" && Number.isFinite(progress)) {
      const pct = Math.max(0, Math.min(100, Math.round(progress * 100)));
      statusEl.textContent = `${text} (${pct}%)`;
    } else {
      statusEl.textContent = text;
    }
  }

  function appendLog(line: string, opts?: { kind?: "info" | "warn" | "error" | "raw"; noStamp?: boolean }) {
    const kind = opts?.kind ?? "info";
    const prefix = opts?.noStamp ? "" : `[${nowStamp()}] `;
    const tag =
      kind === "error" ? "[ERR] " :
      kind === "warn" ? "[WRN] " :
      kind === "raw" ? "" :
      "";

    const text = `${prefix}${tag}${line}`;
    logsEl.textContent = logsEl.textContent
      ? `${logsEl.textContent}\n${text}`
      : text;

    logsEl.scrollTop = logsEl.scrollHeight;
  }

  function renderResults(value: unknown) {
    // Keep it simple and robust for now: pretty JSON/text
    resultsEl.textContent = asPrettyText(value);
  }

  function setRunUiState(active: boolean) {
    runActive = active;
    if (startBtn) startBtn.disabled = !socketOpen || active;
    if (stopBtn) stopBtn.disabled = !socketOpen || !active;
  }

  function setSocketUiState(open: boolean) {
    socketOpen = open;
    if (!open) {
      setRunUiState(false);
    } else {
      setRunUiState(runActive);
    }
  }

  // -----------------------------
  // Config extraction (customize as you add run controls)
  // -----------------------------
  function tryReadRunConfigFromPage(): RunConfig {
    // Option A: hidden JSON textarea/input if you add one later
    const rawCfgEl =
      (document.getElementById("run-config-json") as HTMLTextAreaElement | null) ||
      (document.getElementById("run-config-json") as HTMLInputElement | null);

    if (rawCfgEl && rawCfgEl.value.trim()) {
      try {
        const parsed = JSON.parse(rawCfgEl.value.trim()) as RunConfig;
        return parsed;
      } catch (err) {
        appendLog(`Invalid JSON in #run-config-json: ${err instanceof Error ? err.message : String(err)}`, {
          kind: "error",
        });
        throw err;
      }
    }

    // Option B: button data attribute (data-run-config='{"...": "..."}')
    if (startBtn?.dataset.runConfig) {
      try {
        return JSON.parse(startBtn.dataset.runConfig) as RunConfig;
      } catch (err) {
        appendLog(`Invalid JSON in run-start-btn[data-run-config]: ${err instanceof Error ? err.message : String(err)}`, {
          kind: "error",
        });
        throw err;
      }
    }

    // Option C: fallback MVP config
    // NOTE: Replace with your real protocol fields as needed.
    return {
      // placeholder defaults; adapt to your protocol schema
      // e.g. model: "baseline", dataset: "demo"
    } as RunConfig;
  }

  // -----------------------------
  // Socket send helpers
  // -----------------------------
  function sendJson(msg: unknown) {
    if (ws.readyState !== WebSocket.OPEN) {
      setStatus("socket not connected");
      appendLog("Cannot send message: websocket is not open", { kind: "warn" });
      return;
    }
    ws.send(JSON.stringify(msg));
  }

  function sendRunConfig(config: RunConfig) {
    const msg: ClientToServerMessage = { type: "start_run", config } as ClientToServerMessage;
    sendJson(msg);
    setStatus("starting…");
    setRunUiState(true);
    appendLog("Sent start_run request");
  }

  function sendStopRun() {
    const msg: ClientToServerMessage = { type: "stop_run", run_id: activeRunId ?? undefined } as ClientToServerMessage;
    sendJson(msg);
    setStatus("stopping…");
    appendLog("Sent stop_run request");
  }

  // -----------------------------
  // Server message handling (lenient / protocol-friendly)
  // -----------------------------
  function handleServerMessage(msg: ServerToClientMessage) {
    const m = msg as unknown as JsonRecord;
    const type = String(m.type ?? "unknown");

    switch (type) {
      case "status": {
        const statusText = typeof m.status === "string" ? m.status : "status";
        if (typeof m.run_id === "string") activeRunId = m.run_id;
        const progress =
          typeof m.progress === "number" ? m.progress :
          typeof m.percent === "number" ? (m.percent as number) / 100 :
          undefined;

        setStatus(statusText, progress);

        if (typeof m.state === "string") {
          if (m.state === "queued" || m.state === "running") setRunUiState(true);
          if (m.state === "completed" || m.state === "failed" || m.state === "canceled") setRunUiState(false);
        } else {
          const lower = statusText.toLowerCase();
          if (lower.includes("completed") || lower.includes("finished") || lower.includes("done")) {
            setRunUiState(false);
            activeRunId = null;
          } else if (lower.includes("starting") || lower.includes("running") || lower.includes("accepted") || lower.includes("queued")) {
            setRunUiState(true);
          }
        }

        // Useful extras from backend auth/status bootstrap
        if (typeof m.user_id === "string" || typeof m.role === "string") {
          appendLog(
            `status: ${statusText}` +
              (typeof m.role === "string" ? ` | role=${m.role}` : "") +
              (typeof m.user_id === "string" ? ` | user=${m.user_id}` : ""),
            { kind: "info" }
          );
        }

        break;
      }

      case "log": {
        const line =
          typeof m.line === "string" ? m.line :
          typeof m.log === "string" ? m.log :
          typeof m.message === "string" ? m.message :
          asPrettyText(m);
        appendLog(line, { kind: "raw" });
        break;
      }

      case "logs": {
        if (Array.isArray(m.lines)) {
          for (const line of m.lines) appendLog(String(line), { kind: "raw" });
        } else if (Array.isArray(m.logs)) {
          for (const line of m.logs) appendLog(String(line), { kind: "raw" });
        } else {
          appendLog(asPrettyText(m), { kind: "raw" });
        }
        break;
      }


      case "progress": {
        const statusText = typeof m.phase === "string" ? m.phase : "running";
        const progress = typeof m.progress === "number" ? m.progress : undefined;
        if (typeof m.run_id === "string") activeRunId = m.run_id;
        setStatus(statusText, progress);
        setRunUiState(true);
        break;
      }

      case "phase": {
        if (typeof m.run_id === "string") activeRunId = m.run_id;
        const phase = typeof m.phase === "string" ? m.phase : "phase";
        const status = typeof m.status === "string" ? m.status : phase;
        appendLog(`[${phase}] ${status}`, { kind: "info" });
        break;
      }

      case "artifact-available": {
        const artifact = (m.artifact ?? {}) as JsonRecord;
        const path = typeof artifact.path === "string" ? artifact.path : undefined;
        const uri = typeof artifact.uri === "string" ? artifact.uri : undefined;
        appendLog(`artifact available${path ? `: ${path}` : ""}${uri ? ` (${uri})` : ""}`, { kind: "info" });
        break;
      }

      case "result":
      case "results": {
        // Common payload field variants
        const payload =
          m.result ??
          m.results ??
          m.data ??
          m.payload ??
          m;
        const payloadRecord = (payload && typeof payload === "object") ? (payload as JsonRecord) : null;
        if (payloadRecord && typeof payloadRecord.run_id === "string") activeRunId = String(payloadRecord.run_id);
        renderResults(payload);
        if (payloadRecord && (typeof payloadRecord.manifest_path === "string" || typeof payloadRecord.manifest_uri === "string")) {
          appendLog(`manifest: ${String(payloadRecord.manifest_path ?? payloadRecord.manifest_uri)}`, { kind: "info" });
        }
        appendLog(`Received ${type} payload`);
        break;
      }

      case "error": {
        const detail =
          typeof m.detail === "string" ? m.detail :
          typeof m.error === "string" ? m.error :
          typeof m.message === "string" ? m.message :
          asPrettyText(m);

        setStatus("error");
        appendLog(detail, { kind: "error" });
        setRunUiState(false);
        activeRunId = null;
        break;
      }

      case "run_started": {
        setStatus("run started");
        setRunUiState(true);
        appendLog("Run started");
        break;
      }

      case "run_stopped": {
        setStatus("run stopped");
        setRunUiState(false);
        activeRunId = null;
        appendLog("Run stopped");
        break;
      }

      case "run_completed":
      case "complete":
      case "done": {
        setStatus("run completed");
        setRunUiState(false);
        activeRunId = null;
        renderResults(m.result ?? m.data ?? m);
        appendLog("Run completed");
        break;
      }

      default: {
        // Fallback: try to be useful without crashing on protocol evolution
        if (typeof m.status === "string") {
          const progress = typeof m.progress === "number" ? m.progress : undefined;
          setStatus(m.status, progress);
        }

        if (typeof m.message === "string") {
          appendLog(`[${type}] ${m.message}`);
        } else {
          appendLog(`[${type}] ${asPrettyText(m)}`);
        }
        break;
      }
    }
  }

  // -----------------------------
  // WebSocket lifecycle
  // -----------------------------
  ws.addEventListener("open", () => {
    setSocketUiState(true);
    setStatus("connected");
    appendLog("WebSocket connected");
  });

  ws.addEventListener("message", (ev: MessageEvent) => {
    if (typeof ev.data !== "string") {
      appendLog("Received non-text websocket frame", { kind: "warn" });
      return;
    }

    try {
      const parsed = JSON.parse(ev.data) as ServerToClientMessage;
      handleServerMessage(parsed);
    } catch {
      // If backend occasionally emits plain text logs, still show them.
      appendLog(ev.data, { kind: "raw" });
    }
  });

  ws.addEventListener("error", () => {
    setStatus("socket error");
    appendLog("WebSocket error", { kind: "error" });
  });

  ws.addEventListener("close", (ev) => {
    setSocketUiState(false);

    if (ev.code === 4401) {
      setStatus("unauthorized (session expired?)");
      appendLog("WebSocket unauthorized (4401) — redirecting to login…", { kind: "warn" });
      return;
    }

    setStatus(`disconnected (code ${ev.code})`);
    appendLog(`WebSocket disconnected (code=${ev.code}, reason=${ev.reason || "none"})`, {
      kind: "warn",
    });
  });

  // -----------------------------
  // Button wiring (optional UI elements)
  // -----------------------------
  if (startBtn) {
    startBtn.addEventListener("click", () => {
      try {
        const config = tryReadRunConfigFromPage();
        sendRunConfig(config);
      } catch {
        // Error already logged by config parser
      }
    });
  }

  if (stopBtn) {
    stopBtn.addEventListener("click", () => {
      sendStopRun();
    });
  }

  if (clearLogsBtn) {
    clearLogsBtn.addEventListener("click", () => {
      logsEl.textContent = "";
      appendLog("Logs cleared", { kind: "info" });
    });
  }

  // Initial UI state
  setSocketUiState(ws.readyState === WebSocket.OPEN);
  setStatus(ws.readyState === WebSocket.OPEN ? "connected" : "connecting…");
  setRunUiState(false);
}
