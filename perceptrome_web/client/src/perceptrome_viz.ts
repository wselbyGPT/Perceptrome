// src/perceptrome_viz.ts
import type { RunConfig, ServerToClientMessage, ClientToServerMessage } from "./protocol";

export function setupPerceptromeViz(ws: WebSocket) {
  const statusEl = document.getElementById("status")!;
  const logsEl = document.getElementById("logs")!;
  const resultsEl = document.getElementById("results")!;

  function sendRunConfig(config: RunConfig) {
    const msg: ClientToServerMessage = { type: "start_run", config };
    ws.send(JSON.stringify(msg));
    statusEl.textContent = "starting…";
  }

  function handleServerMessage(msg: ServerToClientMessage) {
    switch (msg.type) {
      case "status":
        statusEl.textContent = `${msg.status} ${msg.progress ? `(${Math.round(msg.progress * 100)}%)` : ""}`;
        break;
      case "log":
        logsEl.textContent += msg.message + "\n";
        break;
      case "result":
        resultsEl.textContent = JSON.stringify(msg.payload, null, 2);
        break;
    }
  }

  return { sendRunConfig, handleServerMessage };
}
