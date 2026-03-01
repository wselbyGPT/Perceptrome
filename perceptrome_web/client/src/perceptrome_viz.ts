// src/perceptrome_viz.ts
import type { ServerToClientMessage, ClientToServerMessage } from "./protocol";

// Local compatibility type for the UI-side run form/config.
// The wire protocol now expects { type: "start_run", command, cwd? } directly.
type RunConfig = {
  command: string;
  cwd?: string;
};

export function setupPerceptromeViz(ws: WebSocket) {
  const statusEl = document.getElementById("status")!;
  const logsEl = document.getElementById("logs")!;
  const resultsEl = document.getElementById("results")!;

  function sendRunConfig(config: RunConfig) {
    const msg: ClientToServerMessage = {
      type: "start_run",
      command: config.command,
      ...(config.cwd ? { cwd: config.cwd } : {}),
    };

    ws.send(JSON.stringify(msg));
    statusEl.textContent = "starting…";
  }

  function handleServerMessage(msg: ServerToClientMessage) {
    switch (msg.type) {
      case "status": {
        const hasProgress =
          typeof (msg as { progress?: number }).progress === "number";
        const progressText = hasProgress
          ? ` (${Math.round(((msg as { progress: number }).progress ?? 0) * 100)}%)`
          : "";

        // Some protocol variants use msg.status, others may use a different field;
        // based on your current code we keep msg.status.
        statusEl.textContent = `${msg.status}${progressText}`;
        break;
      }

      case "log":
        logsEl.textContent += msg.message + "\n";
        break;

      case "result":
        resultsEl.textContent = JSON.stringify(msg.payload, null, 2);
        break;

      default: {
        // Exhaustiveness guard (safe no-op at runtime)
        const _never: never = msg;
        void _never;
      }
    }
  }
  return { sendRunConfig, handleServerMessage };
}
