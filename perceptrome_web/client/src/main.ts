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
                  <input class="field-input" type="text" data-action="project-dir" value="." />
                </label>

                <label class="field">
                  <span class="field-label">stream_config.yaml:</span>
                  <input class="field-input" type="text" data-action="stream-config" value="stream_config.yaml" />
                </label>

                <label class="field">
                  <span class="field-label">Dataset list file:</span>
                  <input class="field-input" type="text" data-action="dataset-list-file" value="config/plasmids_10.txt" />
                </label>

                <label class="field">
                  <span class="field-label">Epochs:</span>
                  <input class="field-input" type="number" data-action="epochs" value="10" />
                </label>

                <label class="field">
                  <span class="field-label">Batch size:</span>
                  <input class="field-input" type="number" data-action="batch-size" value="256" />
                </label>

                <label class="field">
                  <span class="field-label">Learning rate:</span>
                  <input class="field-input" type="number" step="0.000001" data-action="learning-rate" value="0.001" />
                </label>
              </div>
            </section>

            <section class="panel">
              <h2 class="panel-title">Custom dataset builder</h2>

              <div class="form-grid form-grid--two">
                <label class="field">
                  <span class="field-label">Category:</span>
                  <select class="field-input" data-action="quota-category">
                    <option>Plasmids</option>
                    <option>Bacterial genomes</option>
                    <option>Custom</option>
                  </select>
                </label>

                <label class="field">
                  <span class="field-label">Source catalog:</span>
                  <input class="field-input" type="text" data-action="quota-source" value="config/plasmids_10.txt" />
                </label>

                <label class="field">
                  <span class="field-label">Count:</span>
                  <input class="field-input" type="number" data-action="quota-count" value="100" />
                </label>
              </div>

              <label class="checkbox">
                <input type="checkbox" data-action="shuffle-categories" checked />
                <span>Shuffle inside each category</span>
              </label>

              <div class="button-row">
                <button class="btn btn-secondary" type="button" data-action="add-quota-row">Add category quota</button>
                <button class="btn btn-secondary" type="button" data-action="remove-quota-row">Remove selected</button>
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
                  <tbody data-action="quota-table-body">
                    <!-- Empty state, matches Qt view -->
                  </tbody>
                </table>
              </div>

              <div class="form-grid">
                <label class="field">
                  <span class="field-label">Output catalog:</span>
                  <input class="field-input" type="text" data-action="output-catalog" value="config/custom_dataset.txt" />
                </label>
              </div>

              <div class="button-row">
                <button class="btn" type="button" data-action="create-dataset-list">Create dataset list</button>
              </div>

              <div class="button-row button-row--end">
                <button class="btn btn-secondary" type="button">Save config</button>
                <button class="btn" type="button" data-action="go-train">Go to Train tab</button>
              </div>

              <p class="panel-footer-text" data-action="config-status">Loaded saved config (if any).</p>
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
                <select class="field-input" data-action="train-model-type">
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
              <button class="btn btn-secondary" type="button" data-action="train-help">Help</button>
              <button class="btn btn-secondary" type="button" data-action="train-build-command">Build command</button>
              <button class="btn" type="button" data-action="train-start">Start</button>
              <button class="btn btn-danger" type="button" data-action="train-stop" disabled>Stop</button>
            </div>

            <div class="progress-bar" data-action="train-progress">
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
                <select class="field-input" data-action="generate-model">
                  <option>model/checkpoints/latest.pt</option>
                </select>
              </label>

              <label class="field">
                <span class="field-label">Length (bp):</span>
                <input class="field-input" type="number" data-action="generate-length" value="10000" />
              </label>

              <label class="field">
                <span class="field-label">Output FASTA:</span>
                <input class="field-input" type="text" data-action="generate-output" value="generated/novel_plasmid.fasta" />
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
              <button class="btn btn-secondary" type="button" data-action="generate-help">Help</button>
              <button class="btn btn-secondary" type="button" data-action="generate-refresh-models">Refresh models</button>
              <button class="btn btn-secondary" type="button" data-action="generate-build-command">Build command</button>
              <button class="btn" type="button" data-action="generate-start">Start</button>
              <button class="btn btn-danger" type="button" data-action="generate-stop" disabled>Stop</button>
            </div>

            <div class="progress-bar" data-action="generate-progress">
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
                  <input class="field-input" type="text" data-action="view-accession" placeholder="Example: NC_000913.3" />
                </label>

                <label class="field">
                  <span class="field-label">FASTA path:</span>
                  <input class="field-input" type="text" data-action="view-fasta-path" value="generated/novel_plasmid.fasta" />
                </label>
              </div>
            </section>

            <section class="panel">
              <h2 class="panel-title">PDF output</h2>

              <div class="form-grid form-grid--two">
                <label class="field">
                  <span class="field-label">Render mode:</span>
                  <select class="field-input" data-action="view-render-mode">
                    <option value="circular">Circular</option>
                    <option value="linear">Linear</option>
                  </select>
                </label>

                <label class="field">
                  <span class="field-label">Output PDF:</span>
                  <input class="field-input" type="text" data-action="view-output-path" value="generated/circular_genome.pdf" />
                </label>

                <label class="field field--full">
                  <span class="field-label">Title:</span>
                  <input class="field-input" type="text" data-action="view-title" placeholder="Optional title override" />
                </label>
              </div>

              <div class="button-row">
                <button class="btn" type="button" data-action="view-generate-pdf">Generate PDF</button>
                <button class="btn btn-secondary" type="button" data-action="view-open-pdf" disabled>Open / Download PDF</button>
              </div>

              <div class="log-area">
                <div class="log-area__label">PDF generation status will appear here...</div>
                <pre class="log-area__body" data-action="view-log"></pre>
              </div>
            </section>
          </div>
        </section>

        <!-- History -->
        <section class="tab-panel" data-tab-panel="history">
          <section class="panel">
            <div class="panel-header-row">
              <h2 class="panel-title">History</h2>
              <button class="btn btn-secondary" type="button" data-action="clear-history">Clear history</button>
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
                <tbody data-action="history-table-body">
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
type DatasetCategoryQuota = { category: string; source: string; count: number };
type HistoryAction = "start" | "stop" | "success" | "failure" | "dataset-create" | "view-generate";
type HistoryEntry = { time: string; action: HistoryAction; details: string };
type PersistedState = {
  config: Record<string, string | boolean>;
  categoryQuotas: DatasetCategoryQuota[];
  history: HistoryEntry[];
};

type WsMessage =
  | { type: "status"; status: "pending" | "running" | "done" | "error"; progress?: number | null }
  | { type: "log"; message: string }
  | { type: "result"; payload: { exit_code?: number; ok?: boolean; command?: string; output_paths?: string[] } }
  | { type: "model_list"; payload: { ok: boolean; checkpoints: string[]; error?: string } }
  | { type: "create_dataset_result"; payload: { ok: boolean; output_catalog?: string; selected_count?: number; logs?: string[]; error?: string } }
  | { type: "view_log"; message: string }
  | { type: "view_status"; status: "pending" | "running" | "done" | "error" }
  | { type: "view_result"; payload: { ok: boolean; output_path?: string; file_url?: string; status?: string; error?: string } };

const trainCommandEl = document.querySelector<HTMLInputElement>('[data-action="train-command"]');
const generateCommandEl = document.querySelector<HTMLInputElement>('[data-action="generate-command"]');
const projectDirEl = document.querySelector<HTMLInputElement>('[data-action="project-dir"]');
const streamConfigEl = document.querySelector<HTMLInputElement>('[data-action="stream-config"]');
const epochsEl = document.querySelector<HTMLInputElement>('[data-action="epochs"]');
const batchSizeEl = document.querySelector<HTMLInputElement>('[data-action="batch-size"]');
const learningRateEl = document.querySelector<HTMLInputElement>('[data-action="learning-rate"]');
const trainModelTypeEl = document.querySelector<HTMLSelectElement>('[data-action="train-model-type"]');
const trainHelpBtn = document.querySelector<HTMLButtonElement>('[data-action="train-help"]');
const trainBuildCommandBtn = document.querySelector<HTMLButtonElement>('[data-action="train-build-command"]');
const trainStartBtn = document.querySelector<HTMLButtonElement>('[data-action="train-start"]');
const trainStopBtn = document.querySelector<HTMLButtonElement>('[data-action="train-stop"]');
const generateHelpBtn = document.querySelector<HTMLButtonElement>('[data-action="generate-help"]');
const generateRefreshModelsBtn = document.querySelector<HTMLButtonElement>('[data-action="generate-refresh-models"]');
const generateBuildCommandBtn = document.querySelector<HTMLButtonElement>('[data-action="generate-build-command"]');
const generateModelEl = document.querySelector<HTMLSelectElement>('[data-action="generate-model"]');
const generateLengthEl = document.querySelector<HTMLInputElement>('[data-action="generate-length"]');
const generateOutputEl = document.querySelector<HTMLInputElement>('[data-action="generate-output"]');
const generateStartBtn = document.querySelector<HTMLButtonElement>('[data-action="generate-start"]');
const generateStopBtn = document.querySelector<HTMLButtonElement>('[data-action="generate-stop"]');
const trainLogEl = document.querySelector<HTMLPreElement>('[data-action="train-log"]');
const generateLogEl = document.querySelector<HTMLPreElement>('[data-action="generate-log"]');
const trainProgressBarEl = document.querySelector<HTMLElement>('[data-action="train-progress"] .progress-bar__value');
const trainProgressLabelEl = document.querySelector<HTMLElement>('[data-action="train-progress"] .progress-bar__label');
const generateProgressBarEl = document.querySelector<HTMLElement>('[data-action="generate-progress"] .progress-bar__value');
const generateProgressLabelEl = document.querySelector<HTMLElement>('[data-action="generate-progress"] .progress-bar__label');
const datasetListFileEl = document.querySelector<HTMLInputElement>('[data-action="dataset-list-file"]');
const quotaCategoryEl = document.querySelector<HTMLSelectElement>('[data-action="quota-category"]');
const quotaSourceEl = document.querySelector<HTMLInputElement>('[data-action="quota-source"]');
const quotaCountEl = document.querySelector<HTMLInputElement>('[data-action="quota-count"]');
const shuffleCategoriesEl = document.querySelector<HTMLInputElement>('[data-action="shuffle-categories"]');
const addQuotaRowBtn = document.querySelector<HTMLButtonElement>('[data-action="add-quota-row"]');
const removeQuotaRowBtn = document.querySelector<HTMLButtonElement>('[data-action="remove-quota-row"]');
const quotaTableBodyEl = document.querySelector<HTMLTableSectionElement>('[data-action="quota-table-body"]');
const outputCatalogEl = document.querySelector<HTMLInputElement>('[data-action="output-catalog"]');
const createDatasetListBtn = document.querySelector<HTMLButtonElement>('[data-action="create-dataset-list"]');
const configStatusEl = document.querySelector<HTMLParagraphElement>('[data-action="config-status"]');
const viewAccessionEl = document.querySelector<HTMLInputElement>('[data-action="view-accession"]');
const viewFastaPathEl = document.querySelector<HTMLInputElement>('[data-action="view-fasta-path"]');
const viewRenderModeEl = document.querySelector<HTMLSelectElement>('[data-action="view-render-mode"]');
const viewOutputPathEl = document.querySelector<HTMLInputElement>('[data-action="view-output-path"]');
const viewTitleEl = document.querySelector<HTMLInputElement>('[data-action="view-title"]');
const viewGenerateBtn = document.querySelector<HTMLButtonElement>('[data-action="view-generate-pdf"]');
const viewOpenBtn = document.querySelector<HTMLButtonElement>('[data-action="view-open-pdf"]');
const viewLogEl = document.querySelector<HTMLPreElement>('[data-action="view-log"]');
const historyTableBodyEl = document.querySelector<HTMLTableSectionElement>('[data-action="history-table-body"]');
const clearHistoryBtn = document.querySelector<HTMLButtonElement>('[data-action="clear-history"]');

const STORAGE_KEY = "perceptrome.web.state.v1";
const PERSISTED_FIELDS: Record<string, HTMLInputElement | HTMLSelectElement | null> = {
  datasetListFile: datasetListFileEl,
  quotaCategory: quotaCategoryEl,
  quotaSource: quotaSourceEl,
  quotaCount: quotaCountEl,
  shuffleCategories: shuffleCategoriesEl,
  outputCatalog: outputCatalogEl,
  projectDir: projectDirEl,
  streamConfig: streamConfigEl,
  epochs: epochsEl,
  batchSize: batchSizeEl,
  learningRate: learningRateEl,
  trainModelType: trainModelTypeEl,
  generateModel: generateModelEl,
  generateLength: generateLengthEl,
  generateOutput: generateOutputEl,
  trainCommand: trainCommandEl,
  generateCommand: generateCommandEl,
  viewAccession: viewAccessionEl,
  viewFastaPath: viewFastaPathEl,
  viewRenderMode: viewRenderModeEl,
  viewOutputPath: viewOutputPathEl,
  viewTitle: viewTitleEl,
};
const historyEntries: HistoryEntry[] = [];

let ws: WebSocket | null = null;
let activeScope: RunScope | null = null;
let generatedPdfUrl: string | null = null;


function clampProgress(value: number | null | undefined): number {
  if (typeof value !== "number" || Number.isNaN(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function setProgress(scope: RunScope, progress: number | null | undefined) {
  const ratio = clampProgress(progress);
  const pct = `${Math.round(ratio * 100)}%`;
  const bar = scope === "train" ? trainProgressBarEl : generateProgressBarEl;
  const label = scope === "train" ? trainProgressLabelEl : generateProgressLabelEl;
  if (bar) bar.style.width = pct;
  if (label) label.textContent = pct;
}

function shellEscape(value: string): string {
  if (!value) return "''";
  if (/^[a-zA-Z0-9_./:-]+$/.test(value)) return value;
  return `'${value.replace(/'/g, `'\''`)}'`;
}

function buildTrainCommand(): string {
  const config = streamConfigEl?.value?.trim() || "stream_config.yaml";
  const catalog = datasetListFileEl?.value?.trim() || "config/plasmids_10.txt";
  const model = trainModelTypeEl?.value || "mlp";
  const epochs = epochsEl?.value?.trim() || "10";
  const batch = batchSizeEl?.value?.trim() || "256";
  const learningRate = learningRateEl?.value?.trim() || "0.001";

  return [
    "perceptrome",
    "--config",
    shellEscape(config),
    "stream",
    "--catalog",
    shellEscape(catalog),
    "--model-type",
    shellEscape(model),
    "--steps-per-plasmid",
    shellEscape(epochs),
    "--batch-size",
    shellEscape(batch),
    "--learning-rate",
    shellEscape(learningRate),
  ].join(" ");
}

function buildGenerateCommand(): string {
  const config = streamConfigEl?.value?.trim() || "stream_config.yaml";
  const checkpoint = generateModelEl?.value?.trim() || "model/checkpoints/latest.pt";
  const lengthBp = generateLengthEl?.value?.trim() || "10000";
  const output = generateOutputEl?.value?.trim() || "generated/novel_plasmid.fasta";

  return [
    "perceptrome",
    "--config",
    shellEscape(config),
    "generate-plasmid",
    "--checkpoint",
    shellEscape(checkpoint),
    "--length-bp",
    shellEscape(lengthBp),
    "--output",
    shellEscape(output),
  ].join(" ");
}

function requestModelList() {
  const socket = ensureWs();
  const cwd = projectDirEl?.value?.trim() || ".";
  const send = () => socket.send(JSON.stringify({ type: "list_models", cwd }));
  if (socket.readyState === WebSocket.OPEN) send();
  else socket.addEventListener("open", send, { once: true });
}

function applyModelList(checkpoints: string[]) {
  if (!generateModelEl) return;
  generateModelEl.innerHTML = "";
  checkpoints.forEach((checkpoint) => {
    const option = document.createElement("option");
    option.value = checkpoint;
    option.textContent = checkpoint;
    generateModelEl.appendChild(option);
  });
  if (checkpoints.length > 0) generateModelEl.value = checkpoints[0];
}

function setConfigStatus(message: string) {
  if (configStatusEl) configStatusEl.textContent = message;
}

function persistState() {
  const config: PersistedState["config"] = {};
  Object.entries(PERSISTED_FIELDS).forEach(([key, el]) => {
    if (!el) return;
    if (el instanceof HTMLInputElement && el.type === "checkbox") {
      config[key] = el.checked;
      return;
    }
    config[key] = el.value;
  });

  const state: PersistedState = {
    config,
    categoryQuotas: collectCategoryQuotas(),
    history: historyEntries,
  };

  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function appendHistoryRow(entry: HistoryEntry) {
  if (!historyTableBodyEl) return;
  const row = document.createElement("tr");
  row.innerHTML = `<td>${entry.time}</td><td>${entry.action}</td><td>${entry.details}</td>`;
  historyTableBodyEl.appendChild(row);
}

function addHistoryEvent(action: HistoryAction, details: string) {
  const entry: HistoryEntry = {
    time: new Date().toLocaleString(),
    action,
    details,
  };
  historyEntries.push(entry);
  appendHistoryRow(entry);
  persistState();
}

function hydrateState() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) {
    setConfigStatus("No saved config found.");
    return;
  }

  try {
    const parsed = JSON.parse(raw) as Partial<PersistedState>;
    const config = parsed.config ?? {};

    Object.entries(PERSISTED_FIELDS).forEach(([key, el]) => {
      if (!el) return;
      const value = config[key];
      if (value === undefined) return;
      if (el instanceof HTMLInputElement && el.type === "checkbox") {
        el.checked = Boolean(value);
      } else {
        el.value = String(value);
      }
    });

    if (quotaTableBodyEl) {
      quotaTableBodyEl.innerHTML = "";
      (parsed.categoryQuotas ?? []).forEach((quota) => addQuotaRow(quota));
    }

    if (historyTableBodyEl) {
      historyTableBodyEl.innerHTML = "";
    }
    historyEntries.splice(0, historyEntries.length, ...((parsed.history ?? []) as HistoryEntry[]));
    historyEntries.forEach((entry) => appendHistoryRow(entry));

    setConfigStatus("Loaded saved config.");
  } catch {
    setConfigStatus("Saved config could not be loaded.");
  }
}

function getSelectedQuotaRow(): HTMLTableRowElement | null {
  return quotaTableBodyEl?.querySelector<HTMLTableRowElement>('tr[data-action="quota-row"].is-selected') ?? null;
}

function clearQuotaRowSelection() {
  quotaTableBodyEl?.querySelectorAll<HTMLTableRowElement>('tr[data-action="quota-row"]').forEach((row) => {
    row.classList.remove("is-selected");
  });
}

function addQuotaRow(quota: DatasetCategoryQuota) {
  if (!quotaTableBodyEl) return;
  const row = document.createElement("tr");
  row.dataset.action = "quota-row";
  row.dataset.category = quota.category;
  row.dataset.source = quota.source;
  row.dataset.count = String(quota.count);
  row.innerHTML = `<td>${quota.category}</td><td>${quota.source}</td><td>${quota.count}</td>`;
  row.addEventListener("click", () => {
    const alreadySelected = row.classList.contains("is-selected");
    clearQuotaRowSelection();
    if (!alreadySelected) row.classList.add("is-selected");
  });
  quotaTableBodyEl.appendChild(row);
  persistState();
}

function collectCategoryQuotas(): DatasetCategoryQuota[] {
  const rows = quotaTableBodyEl?.querySelectorAll<HTMLTableRowElement>('tr[data-action="quota-row"]') ?? [];
  return Array.from(rows).map((row) => ({
    category: row.dataset.category ?? "",
    source: row.dataset.source ?? "",
    count: Number(row.dataset.count ?? "0"),
  }));
}

function appendLog(scope: RunScope, line: string) {
  const target = scope === "train" ? trainLogEl : generateLogEl;
  if (!target) return;
  target.textContent += `${line}\n`;
  target.scrollTop = target.scrollHeight;
}

function appendViewLog(line: string) {
  if (!viewLogEl) return;
  viewLogEl.textContent += `${line}\n`;
  viewLogEl.scrollTop = viewLogEl.scrollHeight;
}

function setViewOpenState(enabled: boolean) {
  if (viewOpenBtn) viewOpenBtn.disabled = !enabled;
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
    appendViewLog("[web] Connected to backend.");
  });

  ws.addEventListener("close", () => {
    appendLog("train", "[web] Backend connection closed.");
    appendLog("generate", "[web] Backend connection closed.");
    appendViewLog("[web] Backend connection closed.");
    if (activeScope) {
      setRunState(activeScope, false);
      setProgress(activeScope, 0);
      activeScope = null;
    }
  });

  ws.addEventListener("message", (event) => {
    let msg: WsMessage;
    try {
      msg = JSON.parse(String(event.data)) as WsMessage;
    } catch {
      if (!activeScope) return;
      appendLog(activeScope, `[web] Invalid backend message: ${String(event.data)}`);
      return;
    }

    if (msg.type === "create_dataset_result") {
      if (msg.payload.logs?.length) {
        setConfigStatus(msg.payload.logs.join(" | "));
      }
      if (msg.payload.ok && msg.payload.output_catalog) {
        if (outputCatalogEl) outputCatalogEl.value = msg.payload.output_catalog;
        if (datasetListFileEl) datasetListFileEl.value = msg.payload.output_catalog;
        const selectedCount = msg.payload.selected_count ?? 0;
        setConfigStatus(`Created dataset list with ${selectedCount} accessions at ${msg.payload.output_catalog}.`);
        addHistoryEvent("dataset-create", `Created ${msg.payload.output_catalog} (${selectedCount} accessions)`);
      } else {
        setConfigStatus(`Failed to create dataset list: ${msg.payload.error ?? "unknown error"}`);
        addHistoryEvent("failure", `Dataset create failed: ${msg.payload.error ?? "unknown error"}`);
      }
      persistState();
      return;
    }

    if (msg.type === "view_log") {
      appendViewLog(msg.message);
      return;
    }

    if (msg.type === "view_status") {
      if (msg.status === "pending" || msg.status === "done" || msg.status === "error") {
        if (viewGenerateBtn) viewGenerateBtn.disabled = false;
      } else if (viewGenerateBtn) {
        viewGenerateBtn.disabled = true;
      }
      return;
    }

    if (msg.type === "view_result") {
      if (viewGenerateBtn) viewGenerateBtn.disabled = false;
      if (msg.payload.ok) {
        generatedPdfUrl = msg.payload.file_url ?? null;
        setViewOpenState(Boolean(generatedPdfUrl));
        if (msg.payload.output_path && viewOutputPathEl) {
          viewOutputPathEl.value = msg.payload.output_path;
        }
        appendViewLog(`[web] ${msg.payload.status ?? "PDF generated."}`);
        addHistoryEvent("view-generate", `Generated ${msg.payload.output_path ?? "PDF"}`);
      } else {
        generatedPdfUrl = null;
        setViewOpenState(false);
        appendViewLog(`[web] ERROR: ${msg.payload.error ?? "Failed to generate PDF."}`);
        addHistoryEvent("failure", `View generate failed: ${msg.payload.error ?? "unknown error"}`);
      }
      persistState();
      return;
    }

    if (msg.type === "model_list") {
      if (!msg.payload.ok) {
        appendLog("generate", `[web] Failed to list models: ${msg.payload.error ?? "unknown error"}`);
        return;
      }
      applyModelList(msg.payload.checkpoints);
      appendLog("generate", `[web] Loaded ${msg.payload.checkpoints.length} checkpoint(s).`);
      if (msg.payload.checkpoints.length === 0) {
        appendLog("generate", "[web] No checkpoints found. Train first or check project directory.");
      }
      return;
    }

    if (!activeScope) return;

    if (msg.type === "log") {
      appendLog(activeScope, msg.message);
      return;
    }

    if (msg.type === "status") {
      setProgress(activeScope, msg.progress);
      if (msg.status === "done" || msg.status === "error" || msg.status === "pending") {
        if (msg.status === "done") addHistoryEvent("success", `${activeScope}: run completed`);
        if (msg.status === "error") addHistoryEvent("failure", `${activeScope}: run failed`);
        setRunState(activeScope, false);
        if (msg.status === "pending") setProgress(activeScope, 0);
        activeScope = null;
      }
      return;
    }

    if (msg.type === "result") {
      const code = msg.payload.exit_code;
      const outputPaths = msg.payload.output_paths ?? [];
      appendLog(activeScope, `[web] RESULT ${JSON.stringify({ exit_code: code ?? null, output_paths: outputPaths }, null, 2)}`);
      appendLog(activeScope, `[web] Run finished (exit code: ${code ?? "unknown"}).`);
      return;
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
  setProgress(scope, 0);
  appendLog(scope, `[web] Starting: ${command}`);
  addHistoryEvent("start", `${scope}: ${command}`);

  const send = () => {
    socket.send(
      JSON.stringify({
        type: "start_run",
        command,
        cwd: projectDirEl?.value?.trim() || ".",
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
  addHistoryEvent("stop", `${scope}: stop requested`);
}


trainHelpBtn?.addEventListener("click", () => {
  if (!trainCommandEl) return;
  trainCommandEl.value = "perceptrome --help && perceptrome stream --help";
  appendLog("train", "[web] Filled train help command.");
  persistState();
});

trainBuildCommandBtn?.addEventListener("click", () => {
  if (!trainCommandEl) return;
  trainCommandEl.value = buildTrainCommand();
  appendLog("train", "[web] Built train command from config fields.");
  persistState();
});

generateHelpBtn?.addEventListener("click", () => {
  if (!generateCommandEl) return;
  generateCommandEl.value = "perceptrome --help && perceptrome generate-plasmid --help";
  appendLog("generate", "[web] Filled generate help command.");
  persistState();
});

generateBuildCommandBtn?.addEventListener("click", () => {
  if (!generateCommandEl) return;
  generateCommandEl.value = buildGenerateCommand();
  appendLog("generate", "[web] Built generate command from config fields.");
  persistState();
});

generateRefreshModelsBtn?.addEventListener("click", () => {
  appendLog("generate", "[web] Refreshing checkpoints from backend...");
  requestModelList();
});

trainStartBtn?.addEventListener("click", () => startRun("train", trainCommandEl?.value ?? ""));
trainStopBtn?.addEventListener("click", () => stopRun("train"));
generateStartBtn?.addEventListener("click", () => startRun("generate", generateCommandEl?.value ?? ""));
generateStopBtn?.addEventListener("click", () => stopRun("generate"));

addQuotaRowBtn?.addEventListener("click", () => {
  const category = quotaCategoryEl?.value?.trim() ?? "";
  const source = quotaSourceEl?.value?.trim() ?? "";
  const count = Number(quotaCountEl?.value ?? "0");
  if (!category || !source || !Number.isFinite(count) || count <= 0) {
    setConfigStatus("Please provide category, source catalog, and count > 0 before adding a quota row.");
    return;
  }
  addQuotaRow({ category, source, count: Math.trunc(count) });
  setConfigStatus(`Added quota row for ${category}.`);
  persistState();
});

function generateViewPdf() {
  const socket = ensureWs();
  const payload = {
    accession: viewAccessionEl?.value?.trim() ?? "",
    fasta_path: viewFastaPathEl?.value?.trim() ?? "",
    render_mode: viewRenderModeEl?.value ?? "circular",
    title: viewTitleEl?.value?.trim() ?? "",
    output_path: viewOutputPathEl?.value?.trim() ?? "",
  };

  if (!payload.accession && !payload.fasta_path) {
    appendViewLog("[web] ERROR: Provide either accession or FASTA path.");
    return;
  }

  if (viewLogEl) viewLogEl.textContent = "";
  generatedPdfUrl = null;
  setViewOpenState(false);
  if (viewGenerateBtn) viewGenerateBtn.disabled = true;

  const send = () => {
    socket.send(JSON.stringify({ type: "view_generate_pdf", payload }));
    appendViewLog("[web] View PDF generation requested.");
  };

  if (socket.readyState === WebSocket.OPEN) send();
  else socket.addEventListener("open", send, { once: true });
}

function openGeneratedPdf() {
  if (!generatedPdfUrl) {
    appendViewLog("[web] No generated PDF available yet.");
    return;
  }
  window.open(generatedPdfUrl, "_blank", "noopener");
}

viewRenderModeEl?.addEventListener("change", () => {
  const mode = viewRenderModeEl.value;
  const output = viewOutputPathEl?.value?.trim() ?? "";
  const defaults = new Set(["generated/circular_genome.pdf", "generated/linear_genome.pdf", ""]);
  if (viewOutputPathEl && defaults.has(output)) {
    viewOutputPathEl.value = mode === "linear" ? "generated/linear_genome.pdf" : "generated/circular_genome.pdf";
  }
  persistState();
});

viewGenerateBtn?.addEventListener("click", generateViewPdf);
viewOpenBtn?.addEventListener("click", openGeneratedPdf);

removeQuotaRowBtn?.addEventListener("click", () => {
  const selected = getSelectedQuotaRow();
  if (!selected) {
    setConfigStatus("Select a quota row to remove.");
    return;
  }
  selected.remove();
  setConfigStatus("Removed selected quota row.");
  persistState();
});

createDatasetListBtn?.addEventListener("click", () => {
  const socket = ensureWs();
  const categoryQuotas = collectCategoryQuotas();
  if (!categoryQuotas.length) {
    setConfigStatus("Add at least one category quota row before creating a dataset list.");
    return;
  }
  const outputCatalog = outputCatalogEl?.value?.trim() ?? "";
  if (!outputCatalog) {
    setConfigStatus("Output catalog path is required.");
    return;
  }

  const send = () => {
    socket.send(JSON.stringify({
      type: "create_dataset",
      payload: {
        category_quotas: categoryQuotas,
        selected_category_sources: categoryQuotas.map((quota) => ({
          category: quota.category,
          source: quota.source,
        })),
        shuffle_within_category: Boolean(shuffleCategoriesEl?.checked),
        output_catalog: outputCatalog,
      },
    }));
    setConfigStatus("Creating dataset list...");
  };

  if (socket.readyState === WebSocket.OPEN) {
    send();
  } else {
    socket.addEventListener("open", send, { once: true });
  }
});


clearHistoryBtn?.addEventListener("click", () => {
  historyEntries.length = 0;
  if (historyTableBodyEl) historyTableBodyEl.innerHTML = "";
  persistState();
  setConfigStatus("History cleared.");
});

Object.values(PERSISTED_FIELDS).forEach((el) => {
  if (!el) return;
  const eventName = el instanceof HTMLInputElement && el.type === "checkbox" ? "change" : "input";
  el.addEventListener(eventName, () => persistState());
});

setProgress("train", 0);
setProgress("generate", 0);
hydrateState();
requestModelList();
