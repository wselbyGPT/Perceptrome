import {
  getRun,
  getRunLineage,
  getRunsSummary,
  listRuns,
  type LineageNode,
  type RunLineage,
  type RunRecord,
} from "./run_api";
import type { RunConfig, ServerToClientMessage, ClientToServerMessage } from "./protocol";
import { ActiveRunTimeline } from "./perceptrome_viz/active_run_timeline";
import { ArtifactSummary } from "./perceptrome_viz/artifact_summary";
import { asPrettyText, type JsonRecord } from "./perceptrome_viz/dom";
import { MetricsPanel } from "./perceptrome_viz/metrics_panel";
import { RunQueueBoard } from "./perceptrome_viz/run_queue_board";

function getById<T extends Element>(root: ParentNode, id: string): T {
  const element = root.querySelector<T>(`#${id}`);
  if (!element) throw new Error(`Expected element #${id}`);
  return element;
}

function applyRunConfigFromQuery(root: ParentNode): void {
  const params = new URLSearchParams(window.location.search);
  const dataset = params.get("dataset");
  const kind = params.get("kind");
  const configPath = params.get("config_path");

  if (dataset) {
    const datasetEl = root.querySelector<HTMLSelectElement>("#dataset");
    if (datasetEl) {
      if (!Array.from(datasetEl.options).some((o) => o.value === dataset)) {
        const option = document.createElement("option");
        option.value = dataset;
        option.textContent = dataset;
        datasetEl.appendChild(option);
      }
      datasetEl.value = dataset;
    }
  }

  if (kind) {
    const kindEl = root.querySelector<HTMLSelectElement>("#run-kind");
    if (kindEl && Array.from(kindEl.options).some((o) => o.value === kind)) {
      kindEl.value = kind;
    }
  }

  if (configPath) {
    const cfgEl = root.querySelector<HTMLInputElement>("#config-path");
    if (cfgEl) cfgEl.value = configPath;
  }
}

export function setupPerceptromeViz(root: HTMLDivElement, ws: WebSocket) {
  const lineageSvg = getById<SVGSVGElement>(root, "lineage-graph");
  const lineageDetails = getById<HTMLElement>(root, "lineage-details");
  const lineageDepthEl = getById<HTMLInputElement>(root, "lineage-depth");
  const lineageArtifactTypeEl = getById<HTMLInputElement>(root, "lineage-artifact-type");
  const lineageRunStateEl = getById<HTMLSelectElement>(root, "lineage-run-state");
  const historyEl = getById<HTMLElement>(root, "run-history");
  const runForm = getById<HTMLFormElement>(root, "run-form");
  const clearLogsBtn = root.querySelector<HTMLButtonElement>("#clear-logs-btn");
  const logsEl = getById<HTMLElement>(root, "logs");

  const queueBoard = new RunQueueBoard(root);
  const timeline = new ActiveRunTimeline(root);
  const metricsPanel = new MetricsPanel(root);
  const artifactSummary = new ArtifactSummary(root);

  const startBtn = root.querySelector<HTMLButtonElement>("#run-start-btn");
  const stopBtn = root.querySelector<HTMLButtonElement>("#run-stop-btn");
  const refreshHistoryBtn = root.querySelector<HTMLButtonElement>("#refresh-history-btn");
  const refreshLineageBtn = root.querySelector<HTMLButtonElement>("#refresh-lineage-btn");
  const cleanupFns: Array<() => void> = [];

  applyRunConfigFromQuery(root);

  let socketOpen = false;
  let runActive = false;
  let activeRunId: string | null = null;

  async function inspectRun(runId: string) {
    const detail = await getRun(runId);
    artifactSummary.renderResults(detail.result ?? detail);
    artifactSummary.renderArtifacts(detail);
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
      timeline.appendLog(`Failed to load lineage: ${String(err)}`, { kind: "warn" });
      lineageDetails.textContent = `Lineage error: ${String(err)}`;
    }
  }

  function renderHistory(runs: RunRecord[]) {
    historyEl.innerHTML = "";
    for (const run of runs) {
      const item = document.createElement("div");
      item.className = "stack run-row";
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

  async function refreshDashboardBoards() {
    try {
      const summary = await getRunsSummary();
      queueBoard.renderSummary(summary);
      const board = await queueBoard.refresh(async (runId: string) => {
        activeRunId = runId;
        await inspectRun(runId);
        await refreshLineage(runId);
        root.querySelector("#run-drilldown")?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
      if (!activeRunId && board.activeRuns.length > 0) {
        activeRunId = board.activeRuns[0].run_id;
      }
    } catch (err) {
      queueBoard.showBoardError(err);
      timeline.appendLog(`Failed to load dashboard summaries: ${String(err)}`, { kind: "warn" });
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
      await refreshDashboardBoards();
    } catch (err) {
      timeline.appendLog(`Failed to load history: ${String(err)}`, { kind: "warn" });
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
    const el = root.querySelector<HTMLInputElement>(`#${id}`);
    const value = Number(el?.value);
    return Number.isFinite(value) ? value : fallback;
  }

  function readRunConfigFromForm(): RunConfig {
    const kind = getById<HTMLSelectElement>(root, "run-kind").value;
    const configPath = getById<HTMLInputElement>(root, "config-path").value.trim();
    const dataset = getById<HTMLSelectElement>(root, "dataset").value;
    const modelFamily = getById<HTMLSelectElement>(root, "model-family").value;
    if (!configPath) throw new Error("Config path is required");

    return {
      kind,
      config_path: configPath,
      dataset,
      model_family: modelFamily,
      temperature: readNumericInput("temperature", 1.0),
      length_bp: readNumericInput("length-bp", 10000),
      params: { dataset, model_family: modelFamily },
    } as RunConfig;
  }

  function sendJson(msg: unknown) {
    if (ws.readyState !== WebSocket.OPEN) {
      queueBoard.setStatus("socket not connected");
      timeline.appendLog("Cannot send message: websocket is not open", { kind: "warn" });
      return;
    }
    ws.send(JSON.stringify(msg));
  }

  function sendRunConfig(config: RunConfig) {
    const msg: ClientToServerMessage = { type: "start_run", config } as ClientToServerMessage;
    sendJson(msg);
    queueBoard.setStatus("starting…");
    setRunUiState(true);
    timeline.appendLog("Sent start_run request");
  }

  function sendStopRun() {
    const msg: ClientToServerMessage = { type: "stop_run", run_id: activeRunId ?? undefined } as ClientToServerMessage;
    sendJson(msg);
    queueBoard.setStatus("stopping…");
    timeline.appendLog("Sent stop_run request");
  }

  function handleServerMessage(msg: ServerToClientMessage) {
    const m = msg as unknown as JsonRecord;
    const type = String(m.type ?? "unknown");

    switch (type) {
      case "status": {
        const statusText = typeof m.status === "string" ? m.status : "status";
        if (typeof m.run_id === "string") activeRunId = m.run_id;
        const progress = typeof m.progress === "number" ? m.progress : typeof m.percent === "number" ? (m.percent as number) / 100 : undefined;
        queueBoard.setStatus(statusText, progress);
        if (typeof m.state === "string") {
          if (m.state === "queued" || m.state === "running") setRunUiState(true);
          if (m.state === "completed" || m.state === "failed" || m.state === "canceled") setRunUiState(false);
        }
        break;
      }
      case "log":
        timeline.appendLog(typeof m.line === "string" ? m.line : asPrettyText(m), { kind: "raw" });
        break;
      case "progress": {
        const statusText = typeof m.phase === "string" ? m.phase : "running";
        const progress = typeof m.progress === "number" ? m.progress : undefined;
        if (typeof m.run_id === "string") activeRunId = m.run_id;
        queueBoard.setStatus(statusText, progress);
        setRunUiState(true);
        break;
      }
      case "phase": {
        if (typeof m.run_id === "string") activeRunId = m.run_id;
        timeline.appendLog(`[${String(m.phase ?? "phase")}] ${String(m.status ?? "")}`, { kind: "info" });
        timeline.pushTimelineEvent(`${String(m.phase ?? "phase")}: ${String(m.status ?? "")}`);
        break;
      }
      case "metric":
        metricsPanel.pushMetric(String(m.name ?? "metric"), m.value);
        break;
      case "checkpoint": {
        const p = String(m.path ?? "");
        const url = typeof m.download_url === "string" ? m.download_url : undefined;
        artifactSummary.pushArtifact(p, url);
        break;
      }
      case "validation-summary":
        artifactSummary.renderResults({ validation_results: m.summary ?? m });
        break;
      case "artifact-available": {
        const artifact = (m.artifact ?? {}) as JsonRecord;
        const path = typeof artifact.path === "string" ? artifact.path : "";
        const downloadUrl = typeof artifact.download_url === "string" ? artifact.download_url : undefined;
        timeline.appendLog(`artifact available${path ? `: ${path}` : ""}`, { kind: "info" });
        artifactSummary.pushArtifact(path, downloadUrl);
        break;
      }
      case "result":
      case "results": {
        const payload = m.result ?? m.results ?? m.data ?? m.payload ?? m;
        const payloadRecord = payload && typeof payload === "object" ? (payload as JsonRecord) : null;
        if (payloadRecord && typeof payloadRecord.run_id === "string") activeRunId = String(payloadRecord.run_id);
        artifactSummary.renderResults(payload);
        setRunUiState(false);
        void refreshHistory();
        void refreshLineage(activeRunId ?? undefined);
        timeline.appendLog(`Received ${type} payload`);
        break;
      }
      case "error":
        queueBoard.setStatus("error");
        timeline.appendLog(String(m.detail ?? m.message ?? "unknown error"), { kind: "error" });
        setRunUiState(false);
        break;
      case "run_stopped":
        queueBoard.setStatus("run stopped");
        setRunUiState(false);
        void refreshHistory();
        break;
      default:
        timeline.appendLog(`[${type}] ${asPrettyText(m)}`);
    }
  }

  const onOpen = () => {
    setSocketUiState(true);
    queueBoard.setStatus("connected");
    timeline.appendLog("WebSocket connected");
    void refreshHistory();
  };
  const onClose = () => {
    setSocketUiState(false);
    queueBoard.setStatus("disconnected");
    timeline.appendLog("WebSocket disconnected", { kind: "warn" });
  };
  const onMessage = (ev: MessageEvent) => {
    try {
      const parsed = JSON.parse(String(ev.data)) as ServerToClientMessage;
      handleServerMessage(parsed);
    } catch (err) {
      timeline.appendLog(`Malformed server message: ${String(err)}`, { kind: "warn" });
    }
  };

  ws.addEventListener("open", onOpen);
  ws.addEventListener("close", onClose);
  ws.addEventListener("message", onMessage);
  cleanupFns.push(() => ws.removeEventListener("open", onOpen));
  cleanupFns.push(() => ws.removeEventListener("close", onClose));
  cleanupFns.push(() => ws.removeEventListener("message", onMessage));

  const onSubmit = (ev: Event) => {
    ev.preventDefault();
    try {
      sendRunConfig(readRunConfigFromForm());
    } catch (err) {
      timeline.appendLog(String(err), { kind: "error" });
    }
  };
  const onRefreshHistory = () => void refreshHistory();
  const onRefreshLineage = () => void refreshLineage();
  const onClearLogs = () => { logsEl.textContent = ""; };

  runForm.addEventListener("submit", onSubmit);
  refreshHistoryBtn?.addEventListener("click", onRefreshHistory);
  refreshLineageBtn?.addEventListener("click", onRefreshLineage);
  stopBtn?.addEventListener("click", sendStopRun);
  clearLogsBtn?.addEventListener("click", onClearLogs);
  lineageDepthEl.addEventListener("change", onRefreshLineage);
  lineageArtifactTypeEl.addEventListener("change", onRefreshLineage);
  lineageRunStateEl.addEventListener("change", onRefreshLineage);

  cleanupFns.push(() => runForm.removeEventListener("submit", onSubmit));
  cleanupFns.push(() => refreshHistoryBtn?.removeEventListener("click", onRefreshHistory));
  cleanupFns.push(() => refreshLineageBtn?.removeEventListener("click", onRefreshLineage));
  cleanupFns.push(() => stopBtn?.removeEventListener("click", sendStopRun));
  cleanupFns.push(() => clearLogsBtn?.removeEventListener("click", onClearLogs));
  cleanupFns.push(() => lineageDepthEl.removeEventListener("change", onRefreshLineage));
  cleanupFns.push(() => lineageArtifactTypeEl.removeEventListener("change", onRefreshLineage));
  cleanupFns.push(() => lineageRunStateEl.removeEventListener("change", onRefreshLineage));

  setSocketUiState(ws.readyState === WebSocket.OPEN);
  queueBoard.setStatus("connecting…");
  void refreshDashboardBoards();

  return () => {
    cleanupFns.forEach((fn) => fn());
  };
}
