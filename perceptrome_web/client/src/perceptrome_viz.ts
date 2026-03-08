import {
  getRun,
  getRunLineage,
  listRuns,
  type LineageNode,
  type RunLineage,
  type RunRecord,
} from "./run_api";
import type {
  RunConfig,
  ServerToClientMessage,
  ClientToServerMessage,
} from "./protocol";

type JsonRecord = Record<string, unknown>;

function mustEl<T extends Element>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing required element #${id}`);
  return el as unknown as T;
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
  const metricsEl = mustEl<HTMLElement>("metrics");
  const checkpointsEl = mustEl<HTMLElement>("checkpoints");
  const generatedEl = mustEl<HTMLElement>("generated-sequences");
  const validationEl = mustEl<HTMLElement>("validation-results");
  const historyEl = mustEl<HTMLElement>("run-history");
  const runForm = mustEl<HTMLFormElement>("run-form");
  const lineageSvg = mustEl<SVGSVGElement>("lineage-graph");
  const lineageDetails = mustEl<HTMLElement>("lineage-details");
  const lineageDepthEl = mustEl<HTMLInputElement>("lineage-depth");
  const lineageArtifactTypeEl = mustEl<HTMLInputElement>("lineage-artifact-type");
  const lineageRunStateEl = mustEl<HTMLSelectElement>("lineage-run-state");

  const startBtn = document.getElementById("run-start-btn") as HTMLButtonElement | null;
  const stopBtn = document.getElementById("run-stop-btn") as HTMLButtonElement | null;
  const refreshHistoryBtn = document.getElementById("refresh-history-btn") as HTMLButtonElement | null;
  const refreshLineageBtn = document.getElementById("refresh-lineage-btn") as HTMLButtonElement | null;

  let socketOpen = false;
  let runActive = false;
  let activeRunId: string | null = null;

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
    const tag = kind === "error" ? "[ERR] " : kind === "warn" ? "[WRN] " : "";
    const text = `${prefix}${tag}${line}`;
    logsEl.textContent = logsEl.textContent ? `${logsEl.textContent}\n${text}` : text;
    logsEl.scrollTop = logsEl.scrollHeight;
  }

  function renderResults(value: unknown) {
    resultsEl.textContent = asPrettyText(value);
    const payload = (value && typeof value === "object") ? value as JsonRecord : {};
    generatedEl.textContent = asPrettyText(payload.generated_sequences ?? payload.generated ?? []);
    validationEl.textContent = asPrettyText(payload.validation_results ?? payload.validation ?? {});
  }

  async function inspectRun(runId: string) {
    const detail = await getRun(runId);
    renderResults(detail.result ?? detail);
    const links = detail.artifacts.map((a) => `<a href="${a.download_url}">${a.label ?? a.path}</a>`).join("\n");
    checkpointsEl.innerHTML = links || "No artifacts";
  }

  function selectedRunStates(): string[] {
    return Array.from(lineageRunStateEl.selectedOptions).map((opt) => opt.value).filter(Boolean);
  }

  function renderLineage(graph: RunLineage) {
    lineageSvg.innerHTML = "";
    const cols = new Map<number, LineageNode[]>();
    for (const node of graph.nodes) {
      const bucket = cols.get(node.depth) ?? [];
      bucket.push(node);
      cols.set(node.depth, bucket);
    }
    const depthKeys = [...cols.keys()].sort((a, b) => a - b);
    const positions = new Map<string, { x: number; y: number }>();
    const laneWidth = 220;
    const rowHeight = 74;
    const pad = 24;

    for (const depth of depthKeys) {
      const nodes = cols.get(depth) ?? [];
      nodes.sort((a, b) => a.label.localeCompare(b.label));
      nodes.forEach((node, idx) => {
        positions.set(node.id, { x: pad + depth * laneWidth, y: pad + idx * rowHeight });
      });
    }

    const width = Math.max(720, pad * 2 + (Math.max(...depthKeys, 0) + 1) * laneWidth);
    const tallest = Math.max(...Array.from(cols.values()).map((nodes) => nodes.length), 1);
    const height = Math.max(280, pad * 2 + tallest * rowHeight);
    lineageSvg.setAttribute("viewBox", `0 0 ${width} ${height}`);

    for (const edge of graph.edges) {
      const from = positions.get(edge.source);
      const to = positions.get(edge.target);
      if (!from || !to) continue;
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      const sx = from.x + 170;
      const sy = from.y + 20;
      const tx = to.x;
      const ty = to.y + 20;
      const mid = (sx + tx) / 2;
      path.setAttribute("d", `M ${sx} ${sy} C ${mid} ${sy}, ${mid} ${ty}, ${tx} ${ty}`);
      path.setAttribute("class", "lineage-edge");
      lineageSvg.appendChild(path);
    }

    for (const node of graph.nodes) {
      const p = positions.get(node.id);
      if (!p) continue;
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.setAttribute("class", `lineage-node lineage-node--${node.kind}`);
      g.setAttribute("transform", `translate(${p.x}, ${p.y})`);

      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("width", "170");
      rect.setAttribute("height", "40");
      rect.setAttribute("rx", "8");
      rect.setAttribute("class", "lineage-node-rect");

      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", "10");
      text.setAttribute("y", "24");
      text.setAttribute("class", "lineage-node-label");
      text.textContent = node.label.slice(0, 24);

      const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
      const badges = [
        node.hash ? `hash:${node.hash.slice(0, 12)}` : null,
        node.config_snapshot?.sha256 ? `config:${node.config_snapshot.sha256.slice(0, 12)}` : null,
      ].filter(Boolean).join(" | ");
      title.textContent = `${node.kind} ${node.label}${badges ? `\n${badges}` : ""}`;

      g.appendChild(rect);
      g.appendChild(text);
      g.appendChild(title);
      g.addEventListener("click", async () => {
        lineageDetails.textContent = asPrettyText(node);
        if (node.kind === "run" && node.run_id) {
          await inspectRun(node.run_id);
        }
      });
      lineageSvg.appendChild(g);
    }
  }

  async function refreshLineage(runId?: string) {
    const target = runId ?? activeRunId;
    if (!target) {
      lineageDetails.textContent = "Select or run a job to load lineage.";
      lineageSvg.innerHTML = "";
      return;
    }
    try {
      const depth = Number(lineageDepthEl.value || "2");
      const graph = await getRunLineage(target, {
        depth: Number.isFinite(depth) ? depth : 2,
        artifactType: lineageArtifactTypeEl.value.trim(),
        runStates: selectedRunStates(),
      });
      renderLineage(graph);
      lineageDetails.textContent = `Loaded lineage for ${target}: ${graph.nodes.length} nodes / ${graph.edges.length} edges`;
    } catch (err) {
      appendLog(`Failed to load lineage: ${String(err)}`, { kind: "warn" });
      lineageDetails.textContent = `Lineage error: ${String(err)}`;
    }
  }

  function renderHistory(runs: RunRecord[]) {
    historyEl.innerHTML = "";
    for (const run of runs) {
      const item = document.createElement("div");
      item.className = "stack";
      const hdr = document.createElement("div");
      hdr.innerHTML = `<strong>${run.run_id}</strong> [${run.kind}] - ${run.state}`;
      const btn = document.createElement("button");
      btn.className = "btn btn--secondary btn--sm";
      btn.type = "button";
      btn.textContent = "Inspect";
      btn.addEventListener("click", async () => {
        activeRunId = run.run_id;
        await inspectRun(run.run_id);
        await refreshLineage(run.run_id);
      });
      item.appendChild(hdr);
      item.appendChild(btn);
      historyEl.appendChild(item);
    }
  }

  async function refreshHistory() {
    try {
      const runs = await listRuns(30);
      renderHistory(runs);
      if (!activeRunId && runs.length > 0) {
        activeRunId = runs[0].run_id;
        await refreshLineage(activeRunId);
      }
    } catch (err) {
      appendLog(`Failed to load history: ${String(err)}`, { kind: "warn" });
    }
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

  function readNumericInput(id: string, fallback: number): number {
    const el = document.getElementById(id) as HTMLInputElement | null;
    const value = Number(el?.value);
    return Number.isFinite(value) ? value : fallback;
  }

  function readRunConfigFromForm(): RunConfig {
    const kind = (document.getElementById("run-kind") as HTMLSelectElement).value;
    const configPath = (document.getElementById("config-path") as HTMLInputElement).value.trim();
    const dataset = (document.getElementById("dataset") as HTMLSelectElement).value;
    const modelFamily = (document.getElementById("model-family") as HTMLSelectElement).value;
    if (!configPath) throw new Error("Config path is required");

    return {
      kind,
      config_path: configPath,
      dataset,
      model_family: modelFamily,
      temperature: readNumericInput("temperature", 1.0),
      length_bp: readNumericInput("length-bp", 10000),
      params: {
        dataset,
        model_family: modelFamily,
      },
    } as RunConfig;
  }

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
        }
        break;
      }

      case "log":
        appendLog(typeof m.line === "string" ? m.line : asPrettyText(m), { kind: "raw" });
        break;

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
        appendLog(`[${String(m.phase ?? "phase")}] ${String(m.status ?? "")}`, { kind: "info" });
        break;
      }

      case "metric": {
        const line = `${String(m.name ?? "metric")}: ${String(m.value ?? "")}`;
        metricsEl.textContent = metricsEl.textContent ? `${metricsEl.textContent}\n${line}` : line;
        break;
      }

      case "checkpoint": {
        const p = String(m.path ?? "");
        const url = typeof m.download_url === "string" ? m.download_url : "";
        checkpointsEl.innerHTML += `${url ? `<a href="${url}">${p}</a>` : p}<br/>`;
        break;
      }

      case "validation-summary":
        validationEl.textContent = asPrettyText(m.summary ?? m);
        break;

      case "artifact-available": {
        const artifact = (m.artifact ?? {}) as JsonRecord;
        const path = typeof artifact.path === "string" ? artifact.path : undefined;
        const downloadUrl = typeof artifact.download_url === "string" ? artifact.download_url : undefined;
        appendLog(`artifact available${path ? `: ${path}` : ""}`, { kind: "info" });
        if (downloadUrl) {
          checkpointsEl.innerHTML += `<a href="${downloadUrl}">${path ?? downloadUrl}</a><br/>`;
        }
        break;
      }

      case "result":
      case "results": {
        const payload = m.result ?? m.results ?? m.data ?? m.payload ?? m;
        const payloadRecord = (payload && typeof payload === "object") ? (payload as JsonRecord) : null;
        if (payloadRecord && typeof payloadRecord.run_id === "string") activeRunId = String(payloadRecord.run_id);
        renderResults(payload);
        setRunUiState(false);
        void refreshHistory();
        void refreshLineage(activeRunId ?? undefined);
        appendLog(`Received ${type} payload`);
        break;
      }

      case "error":
        setStatus("error");
        appendLog(String(m.detail ?? m.message ?? "unknown error"), { kind: "error" });
        setRunUiState(false);
        break;

      case "run_stopped":
        setStatus("run stopped");
        setRunUiState(false);
        void refreshHistory();
        break;

      default:
        appendLog(`[${type}] ${asPrettyText(m)}`);
    }
  }

  ws.addEventListener("open", () => {
    setSocketUiState(true);
    setStatus("connected");
    appendLog("WebSocket connected");
    void refreshHistory();
  });

  ws.addEventListener("close", () => {
    setSocketUiState(false);
    setStatus("disconnected");
    appendLog("WebSocket disconnected", { kind: "warn" });
  });

  ws.addEventListener("message", (ev) => {
    try {
      const parsed = JSON.parse(String(ev.data)) as ServerToClientMessage;
      handleServerMessage(parsed);
    } catch (err) {
      appendLog(`Malformed server message: ${String(err)}`, { kind: "warn" });
    }
  });

  runForm.addEventListener("submit", (ev) => {
    ev.preventDefault();
    try {
      sendRunConfig(readRunConfigFromForm());
    } catch (err) {
      appendLog(String(err), { kind: "error" });
    }
  });

  stopBtn?.addEventListener("click", () => sendStopRun());
  refreshHistoryBtn?.addEventListener("click", () => void refreshHistory());
  refreshLineageBtn?.addEventListener("click", () => void refreshLineage());
  lineageDepthEl.addEventListener("change", () => void refreshLineage());
  lineageArtifactTypeEl.addEventListener("change", () => void refreshLineage());
  lineageRunStateEl.addEventListener("change", () => void refreshLineage());

  setSocketUiState(ws.readyState === WebSocket.OPEN);
  setStatus("connecting…");
}
