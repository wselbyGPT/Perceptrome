import { U } from "./utils.js";
import { api } from "./api.js";
import { state } from "./state.js";
import { TRAIN_ENDPOINTS } from "./config.js";
import { Toasts } from "./toasts.js";

function parseMaybeNumber(v) {
  if (v === null || v === undefined) return undefined;
  const s = String(v).trim();
  if (!s) return undefined;
  const n = Number(s);
  return Number.isFinite(n) ? n : undefined;
}

function renderTrainJobs(obj) {
  const host = U.$("#trainJobs");
  if (!host) return;

  const runs = obj?.runs || obj?.jobs || obj?.items || obj?.history || (Array.isArray(obj) ? obj : null);
  host.innerHTML = "";
  if (!runs || !Array.isArray(runs) || runs.length === 0) {
    host.appendChild(U.el("div", { class: "pad muted", text: "No runs/jobs found in status." }));
    return;
  }

  for (const r of runs.slice(0, 50)) {
    const id = r.id ?? r.run_id ?? r.job_id ?? r.name ?? "run";
    const state0 = r.state ?? r.status ?? r.phase ?? "—";
    const started = r.started ?? r.start_time ?? r.created ?? "";
    host.appendChild(
      U.el("div", { class: "item" },
        U.el("div", { class: "l" }, U.el("div", { class: "mono", text: String(id) }), U.el("div", { class: "muted", text: started ? `started: ${U.fmtIso(started)}` : "started: —" })),
        U.el("div", { class: "r" }, U.el("div", { class: "muted", text: String(state0) }))
      )
    );
  }
}

function renderTrainStatus(obj) {
  const host = U.$("#trainStatus");
  if (!host) return;
  if (!obj) { host.innerHTML = `<div class="muted">No status.</div>`; return; }

  const active = obj.active ?? obj.running ?? obj.is_running ?? obj.busy;
  const step = obj.step ?? obj.global_step ?? obj.iter ?? obj.iteration;
  const loss = obj.loss ?? obj.train_loss ?? obj.cur_loss;
  const lr = obj.lr ?? obj.learning_rate;
  const device = obj.device ?? obj.accelerator;

  host.innerHTML = `
    <div class="pretty">
      <div class="title"><div class="h">Training Status</div><div class="sub">best-effort</div></div>
      <div class="pretty-grid">
        <div class="box">
          <div class="row"><span class="k">active</span><span class="v">${U.esc(active === undefined ? "—" : String(!!active))}</span></div>
          <div class="row"><span class="k">step</span><span class="v">${U.esc(step === undefined ? "—" : String(step))}</span></div>
          <div class="row"><span class="k">loss</span><span class="v">${U.esc(loss === undefined ? "—" : String(loss))}</span></div>
          <div class="row"><span class="k">lr</span><span class="v">${U.esc(lr === undefined ? "—" : String(lr))}</span></div>
          <div class="row"><span class="k">device</span><span class="v">${U.esc(device === undefined ? "—" : String(device))}</span></div>
        </div>
        <div class="box">
          <div class="row"><span class="k">raw keys</span><span class="v">${U.esc(Object.keys(obj).slice(0, 10).join(", ") || "—")}</span></div>
          <details class="rawjson"><summary>Show raw JSON</summary><pre>${U.esc(JSON.stringify(obj, null, 2))}</pre></details>
        </div>
      </div>
    </div>
  `;
}

export async function refreshTraining() {
  try {
    const r = await api.tryPaths("GET", TRAIN_ENDPOINTS.status);
    const obj = r.json ?? (r.text ? { text: r.text } : null);
    renderTrainStatus(obj);
    renderTrainJobs(obj);
  } catch (e) {
    const host = U.$("#trainStatus");
    if (host) host.innerHTML = `<div class="muted">No training status endpoint found yet.</div><div class="muted" style="margin-top:8px">${U.esc(e.message)}</div>`;
  }

  try {
    const r = await api.tryPaths("GET", TRAIN_ENDPOINTS.logs);
    const t = r.text != null ? r.text : r.json != null ? JSON.stringify(r.json, null, 2) : "";
    const pre = U.$("#trainLogs");
    if (pre) pre.textContent = t || "(empty)";
  } catch {
    const pre = U.$("#trainLogs");
    if (pre) pre.textContent = "(no logs endpoint found)";
  }
}

export async function startTraining() {
  const payload = {
    catalog: (U.$("#trainCatalog")?.value || "").trim() || undefined,
    kind: (U.$("#trainKind")?.value || "").trim() || undefined,
    run_name: (U.$("#trainRunName")?.value || "").trim() || undefined,
    steps: parseMaybeNumber(U.$("#trainSteps")?.value),
    epochs: parseMaybeNumber(U.$("#trainSteps")?.value),
    batch_size: parseMaybeNumber(U.$("#trainBatch")?.value),
    lr: parseMaybeNumber(U.$("#trainLr")?.value),
    learning_rate: parseMaybeNumber(U.$("#trainLr")?.value),
    device: (U.$("#trainDevice")?.value || "").trim() || undefined,
  };
  for (const k of Object.keys(payload)) if (payload[k] === undefined) delete payload[k];

  try {
    await api.tryPaths("POST", TRAIN_ENDPOINTS.start, payload);
    Toasts.toast("Training started (best-effort).", "good", 2800);
    await refreshTraining().catch(() => {});
  } catch {
    Toasts.toast("No training start endpoint found yet.", "warn", 3600);
  }
}

export async function stopTraining() {
  const runId = (U.$("#trainRunId")?.value || "").trim() || undefined;
  const payload = runId ? { run_id: runId, id: runId } : {};
  try {
    await api.tryPaths("POST", TRAIN_ENDPOINTS.stop, payload);
    Toasts.toast("Stop requested.", "good", 2400);
    await refreshTraining().catch(() => {});
  } catch {
    Toasts.toast("No training stop endpoint found yet.", "warn", 3600);
  }
}

export function syncTrainingCatalogOptions() {
  const sel = U.$("#trainCatalog");
  if (!sel) return;
  const cur = sel.value;
  sel.innerHTML = "";
  sel.appendChild(U.el("option", { value: "", text: "(select dataset catalog)" }));
  for (const c of state.catalogs || []) {
    const name = c.catalog || c.name || c.id || "";
    if (name) sel.appendChild(U.el("option", { value: name, text: name }));
  }
  if (cur && Array.from(sel.options).some(o => o.value === cur)) sel.value = cur;
}
