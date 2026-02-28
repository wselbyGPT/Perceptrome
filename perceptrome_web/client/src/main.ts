import "./styles.css";

type TabId = "config" | "train" | "generate" | "view" | "history";

const app = document.querySelector<HTMLDivElement>("#app")!;
app.innerHTML = `
  <div class="app-root">
    <header class="app-header">
      <div class="app-title">perceptrome</div>
    </header>

    <div class="app-body">
      <nav class="tabs" aria-label="Main">
        <button class="tab tab--active" data-tab="config">Home / Config</button>
        <button class="tab" data-tab="train">Train</button>
        <button class="tab" data-tab="generate">Generate</button>
        <button class="tab" data-tab="view">View</button>
        <button class="tab" data-tab="history">History</button>
      </nav>

      <main class="tab-panels">
        <!-- Home / Config -->
        <section class="tab-panel tab-panel--active" data-tab-panel="config">
          <div class="panel-group">
            <section class="panel">
              <h2 class="panel-title">Project settings</h2>
              <div class="form-grid">
                <label class="field">
                  <span class="field-label">Project dir:</span>
                  <input class="field-input" type="text" value="." />
                </label>

                <label class="field">
                  <span class="field-label">stream_config.yaml:</span>
                  <input class="field-input" type="text" value="stream_config.yaml" />
                </label>

                <label class="field">
                  <span class="field-label">Dataset list file:</span>
                  <input class="field-input" type="text" value="config/plasmids_10.txt" />
                </label>

                <label class="field">
                  <span class="field-label">Epochs:</span>
                  <input class="field-input" type="number" value="10" />
                </label>

                <label class="field">
                  <span class="field-label">Batch size:</span>
                  <input class="field-input" type="number" value="256" />
                </label>

                <label class="field">
                  <span class="field-label">Learning rate:</span>
                  <input class="field-input" type="number" step="0.000001" value="0.001" />
                </label>
              </div>
            </section>

            <section class="panel">
              <h2 class="panel-title">Custom dataset builder</h2>

              <div class="form-grid form-grid--two">
                <label class="field">
                  <span class="field-label">Category:</span>
                  <select class="field-input">
                    <option>Plasmids</option>
                    <option>Bacterial genomes</option>
                    <option>Custom</option>
                  </select>
                </label>

                <label class="field">
                  <span class="field-label">Count:</span>
                  <input class="field-input" type="number" value="100" />
                </label>
              </div>

              <label class="checkbox">
                <input type="checkbox" checked />
                <span>Shuffle inside each category</span>
              </label>

              <div class="button-row">
                <button class="btn btn-secondary" type="button">Add category quota</button>
                <button class="btn btn-secondary" type="button">Remove selected</button>
              </div>

              <div class="table-wrapper">
                <table class="table">
                  <thead>
                    <tr>
                      <th>Category</th>
                      <th>Source</th>
                      <th>Count</th>
                    </tr>
                  </thead>
                  <tbody>
                    <!-- Empty state, matches Qt view -->
                  </tbody>
                </table>
              </div>

              <div class="form-grid">
                <label class="field">
                  <span class="field-label">Output catalog:</span>
                  <input class="field-input" type="text" value="config/custom_dataset.txt" />
                </label>
              </div>

              <div class="button-row">
                <button class="btn" type="button">Create dataset list</button>
              </div>

              <div class="button-row button-row--end">
                <button class="btn btn-secondary" type="button">Save config</button>
                <button class="btn" type="button" data-action="go-train">Go to Train tab</button>
              </div>

              <p class="panel-footer-text">Loaded saved config (if any).</p>
            </section>
          </div>
        </section>

        <!-- Train -->
        <section class="tab-panel" data-tab-panel="train">
          <section class="panel">
            <h2 class="panel-title">Training setup</h2>

            <div class="form-grid">
              <label class="field">
                <span class="field-label">Neural network:</span>
                <select class="field-input">
                  <option>mlp</option>
                  <option>cnn</option>
                  <option>transformer</option>
                </select>
              </label>

              <label class="field field--full">
                <span class="field-label">Training command:</span>
                <input
                  class="field-input"
                  type="text"
                  data-action="train-command"
                  value="perceptrome --config config/stream_config.yaml stream --catalog config/plasmids_10.txt --model-type mlp --steps-per-plasmid 10 --batch-size 256"
                />
              </label>
            </div>

            <div class="button-row">
              <button class="btn btn-secondary" type="button">Help</button>
              <button class="btn btn-secondary" type="button">Build command</button>
              <button class="btn" type="button" data-action="train-start">Start</button>
              <button class="btn btn-danger" type="button" data-action="train-stop" disabled>Stop</button>
            </div>

            <div class="progress-bar">
              <div class="progress-bar__track">
                <div class="progress-bar__value" style="width: 0%;"></div>
              </div>
              <div class="progress-bar__label">0%</div>
            </div>

            <div class="log-area">
              <div class="log-area__label">Live training output will appear here...</div>
              <pre class="log-area__body" data-action="train-log"></pre>
            </div>
          </section>
        </section>

        <!-- Generate -->
        <section class="tab-panel" data-tab-panel="generate">
          <section class="panel">
            <h2 class="panel-title">Generation setup</h2>

            <div class="form-grid">
              <label class="field">
                <span class="field-label">Trained model:</span>
                <select class="field-input">
                  <option>model/checkpoints/latest.pt</option>
                </select>
              </label>

              <label class="field field--full">
                <span class="field-label">Generate command:</span>
                <input
                  class="field-input"
                  type="text"
                  data-action="generate-command"
                  value="perceptrome --config config/stream_config.yaml generate-plasmid --length-bp 10000 --output generated/novel_plasmid.fasta"
                />
              </label>
            </div>

            <div class="button-row">
              <button class="btn btn-secondary" type="button">Help</button>
              <button class="btn btn-secondary" type="button">Refresh models</button>
              <button class="btn btn-secondary" type="button">Build command</button>
              <button class="btn" type="button" data-action="generate-start">Start</button>
              <button class="btn btn-danger" type="button" data-action="generate-stop" disabled>Stop</button>
            </div>

            <div class="progress-bar">
              <div class="progress-bar__track">
                <div class="progress-bar__value" style="width: 0%;"></div>
              </div>
              <div class="progress-bar__label">0%</div>
            </div>

            <div class="log-area">
              <div class="log-area__label">Live generate output will appear here...</div>
              <pre class="log-area__body" data-action="generate-log"></pre>
            </div>
          </section>
        </section>

        <!-- View -->
        <section class="tab-panel" data-tab-panel="view">
          <div class="panel-group">
            <section class="panel">
              <h2 class="panel-title">Genome source</h2>

              <div class="form-grid">
                <label class="field">
                  <span class="field-label">Genome accession:</span>
                  <input class="field-input" type="text" placeholder="Example: NC_000913.3" />
                </label>

                <label class="field">
                  <span class="field-label">FASTA path:</span>
                  <input class="field-input" type="text" value="generated/novel_plasmid.fasta" />
                </label>
              </div>
            </section>

            <section class="panel">
              <h2 class="panel-title">PDF output</h2>

              <div class="form-grid form-grid--two">
                <label class="field">
                  <span class="field-label">Render mode:</span>
                  <select class="field-input">
                    <option>Circular</option>
                    <option>Linear</option>
                  </select>
                </label>

                <label class="field">
                  <span class="field-label">Output PDF:</span>
                  <input class="field-input" type="text" value="generated/circular_genome.pdf" />
                </label>

                <label class="field field--full">
                  <span class="field-label">Title:</span>
                  <input class="field-input" type="text" placeholder="Optional title override" />
                </label>
              </div>

              <div class="button-row">
                <button class="btn" type="button">Generate PDF</button>
                <button class="btn btn-secondary" type="button">Open PDF Window</button>
              </div>

              <div class="log-area">
                <div class="log-area__label">PDF generation status will appear here...</div>
                <pre class="log-area__body"></pre>
              </div>
            </section>
          </div>
        </section>

        <!-- History -->
        <section class="tab-panel" data-tab-panel="history">
          <section class="panel">
            <div class="panel-header-row">
              <h2 class="panel-title">History</h2>
              <button class="btn btn-secondary" type="button">Clear history</button>
            </div>

            <div class="table-wrapper">
              <table class="table">
                <thead>
                  <tr>
                    <th style="width: 20%;">Time</th>
                    <th style="width: 20%;">Action</th>
                    <th>Details</th>
                  </tr>
                </thead>
                <tbody>
                  <!-- Empty, matches Qt screenshot -->
                </tbody>
              </table>
            </div>
          </section>
        </section>
      </main>
    </div>
  </div>
`;

// --- tab behaviour ---------------------------------------------------------

const tabs = Array.from(document.querySelectorAll<HTMLButtonElement>(".tab"));
const panels = Array.from(
  document.querySelectorAll<HTMLElement>(".tab-panel")
);

function setActiveTab(id: TabId) {
  tabs.forEach((tab) => {
    const isActive = tab.dataset.tab === id;
    tab.classList.toggle("tab--active", isActive);
  });

  panels.forEach((panel) => {
    const isActive = panel.dataset.tabPanel === id;
    panel.classList.toggle("tab-panel--active", isActive);
  });
}

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const id = tab.dataset.tab as TabId;
    setActiveTab(id);
  });
});

// "Go to Train tab" button on Config page
const goTrainBtn = document.querySelector<HTMLButtonElement>(
  '[data-action="go-train"]'
);
if (goTrainBtn) {
  goTrainBtn.addEventListener("click", () => setActiveTab("train"));
}

// --- backend wiring --------------------------------------------------------

type RunScope = "train" | "generate";

type WsMessage =
  | { type: "status"; status: "pending" | "running" | "done" | "error"; progress?: number | null }
  | { type: "log"; message: string }
  | { type: "result"; payload: { exit_code?: number; ok?: boolean; command?: string } };

const trainCommandEl = document.querySelector<HTMLInputElement>('[data-action="train-command"]');
const generateCommandEl = document.querySelector<HTMLInputElement>('[data-action="generate-command"]');
const trainStartBtn = document.querySelector<HTMLButtonElement>('[data-action="train-start"]');
const trainStopBtn = document.querySelector<HTMLButtonElement>('[data-action="train-stop"]');
const generateStartBtn = document.querySelector<HTMLButtonElement>('[data-action="generate-start"]');
const generateStopBtn = document.querySelector<HTMLButtonElement>('[data-action="generate-stop"]');
const trainLogEl = document.querySelector<HTMLPreElement>('[data-action="train-log"]');
const generateLogEl = document.querySelector<HTMLPreElement>('[data-action="generate-log"]');

let ws: WebSocket | null = null;
let activeScope: RunScope | null = null;

function appendLog(scope: RunScope, line: string) {
  const target = scope === "train" ? trainLogEl : generateLogEl;
  if (!target) return;
  target.textContent += `${line}\n`;
  target.scrollTop = target.scrollHeight;
}

function setRunState(scope: RunScope, running: boolean) {
  if (scope === "train") {
    if (trainStartBtn) trainStartBtn.disabled = running;
    if (trainStopBtn) trainStopBtn.disabled = !running;
  } else {
    if (generateStartBtn) generateStartBtn.disabled = running;
    if (generateStopBtn) generateStopBtn.disabled = !running;
  }
}

function ensureWs(): WebSocket {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return ws;
  }

  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${protocol}://${window.location.host}/ws`);

  ws.addEventListener("open", () => {
    appendLog("train", "[web] Connected to backend.");
    appendLog("generate", "[web] Connected to backend.");
  });

  ws.addEventListener("close", () => {
    appendLog("train", "[web] Backend connection closed.");
    appendLog("generate", "[web] Backend connection closed.");
    if (activeScope) {
      setRunState(activeScope, false);
      activeScope = null;
    }
  });

  ws.addEventListener("message", (event) => {
    if (!activeScope) return;
    let msg: WsMessage;
    try {
      msg = JSON.parse(String(event.data)) as WsMessage;
    } catch {
      appendLog(activeScope, `[web] Invalid backend message: ${String(event.data)}`);
      return;
    }

    if (msg.type === "log") {
      appendLog(activeScope, msg.message);
      return;
    }

    if (msg.type === "status") {
      if (msg.status === "done" || msg.status === "error" || msg.status === "pending") {
        setRunState(activeScope, false);
        activeScope = null;
      }
      return;
    }

    if (msg.type === "result") {
      const code = msg.payload.exit_code;
      appendLog(activeScope, `[web] Run finished (exit code: ${code ?? "unknown"}).`);
    }
  });

  return ws;
}

function startRun(scope: RunScope, command: string) {
  const socket = ensureWs();
  if (!command.trim()) {
    appendLog(scope, "[web] Command is empty.");
    return;
  }
  if (activeScope) {
    appendLog(scope, "[web] Another run is active. Stop it before starting a new one.");
    return;
  }

  activeScope = scope;
  setRunState(scope, true);
  appendLog(scope, `[web] Starting: ${command}`);

  const send = () => {
    socket.send(
      JSON.stringify({
        type: "start_run",
        command,
        cwd: ".",
      }),
    );
  };

  if (socket.readyState === WebSocket.OPEN) {
    send();
  } else {
    socket.addEventListener("open", send, { once: true });
  }
}

function stopRun(scope: RunScope) {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    appendLog(scope, "[web] Backend is not connected.");
    return;
  }
  if (activeScope !== scope) {
    appendLog(scope, "[web] No active run in this tab.");
    return;
  }

  ws.send(JSON.stringify({ type: "stop_run" }));
  appendLog(scope, "[web] Stop requested.");
}

trainStartBtn?.addEventListener("click", () => startRun("train", trainCommandEl?.value ?? ""));
trainStopBtn?.addEventListener("click", () => stopRun("train"));
generateStartBtn?.addEventListener("click", () => startRun("generate", generateCommandEl?.value ?? ""));
generateStopBtn?.addEventListener("click", () => stopRun("generate"));
