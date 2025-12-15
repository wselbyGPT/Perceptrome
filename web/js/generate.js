import { U } from "./utils.js";
import { api } from "./api.js";
import { state } from "./state.js";
import { LS } from "./config.js";
import { Store } from "./store.js";
import { Toasts } from "./toasts.js";

/* pretty blocks are inlined minimal here (same output as before) */
function meterHtml(value0to100, leftLabel, rightLabel) {
  const w = U.clamp(value0to100 ?? 0, 0, 100);
  return `
    <div class="meter"><div class="fill" style="width:${w}%"></div></div>
    <div class="meter-label"><span>${U.esc(leftLabel)}</span><span>${U.esc(rightLabel)}</span></div>
  `;
}
function pctBarHtml(pct0to100) {
  const p = U.clamp(pct0to100 ?? 0, 0, 100);
  return `
    <div class="pctbar"><div class="zone"></div><div class="mark" style="left: calc(${p}% - 1px)"></div></div>
    <div class="meter-label"><span>0</span><span>25</span><span>50</span><span>75</span><span>100</span></div>
  `;
}
function badge(cls, k, v) { return `<div class="bx ${cls}"><span class="k">${U.esc(k)}</span><span class="v">${U.esc(v)}</span></div>`; }

function prettyValidateHTML(r) {
  const kind = r.kind || "—";
  const file = r.file?.name || r.file || "—";
  const len = Number(r.length ?? 0).toLocaleString();

  if (kind === "protein") {
    const x = U.pct(r.x_pct);
    const mr = r.max_run || {};
    const run = mr.run ?? 0;
    const base = mr.base ?? "—";
    const clsX = x != null && x <= 2 ? "good" : x != null && x <= 10 ? "warn" : "bad";
    const clsRun = run <= 10 ? "good" : run <= 25 ? "warn" : "bad";
    return `
      <div class="pretty">
        <div class="title"><div class="h">Validate</div><div class="sub">${U.esc(file)} • protein</div></div>
        <div class="pretty-grid">
          <div class="box">
            <div class="row"><span class="k">length</span><span class="v">${U.esc(len)}</span></div>
            <div class="row"><span class="k">X%</span><span class="v">${x == null ? "—" : x.toFixed(2) + "%"}</span></div>
            ${meterHtml(100 - U.clamp(x ?? 0, 0, 100), "more clean", "more X")}
            <div class="badges">
              ${badge(clsX, "X%", x == null ? "—" : x.toFixed(2) + "%")}
              ${badge(clsRun, "max-run", `${base}×${run}`)}
            </div>
          </div>
          <div class="box">
            <div class="row"><span class="k">homopolymer max-run</span><span class="v">${U.esc(`${base}×${run}`)}</span></div>
          </div>
        </div>
        <details class="rawjson"><summary>Show raw JSON</summary><pre>${U.esc(JSON.stringify(r, null, 2))}</pre></details>
      </div>
    `;
  }

  const gc = U.pct(r.gc_pct);
  const np = U.pct(r.n_pct);
  const inv = U.pct(r.invalid_pct);
  const mr = r.max_run || {};
  const run = mr.run ?? 0;
  const base = mr.base ?? "—";
  const clsN = np != null && np <= 1 ? "good" : np != null && np <= 5 ? "warn" : "bad";
  const clsInv = inv != null && inv <= 0.1 ? "good" : inv != null && inv <= 1 ? "warn" : "bad";
  const clsRun = run <= 12 ? "good" : run <= 30 ? "warn" : "bad";

  return `
    <div class="pretty">
      <div class="title"><div class="h">Validate</div><div class="sub">${U.esc(file)} • genome</div></div>
      <div class="pretty-grid">
        <div class="box">
          <div class="row"><span class="k">length</span><span class="v">${U.esc(len)} bp</span></div>
          <div class="row"><span class="k">GC%</span><span class="v">${gc == null ? "—" : gc.toFixed(2) + "%"}</span></div>
          <div class="row"><span class="k">N%</span><span class="v">${np == null ? "—" : np.toFixed(2) + "%"}</span></div>
          <div class="row"><span class="k">invalid%</span><span class="v">${inv == null ? "—" : inv.toFixed(3) + "%"}</span></div>
          <div class="badges">
            ${badge(clsN, "N%", np == null ? "—" : np.toFixed(2) + "%")}
            ${badge(clsInv, "invalid", inv == null ? "—" : inv.toFixed(3) + "%")}
            ${badge(clsRun, "max-run", `${base}×${run}`)}
          </div>
        </div>
        <div class="box">
          <div class="row"><span class="k">notes</span><span class="v">ORF details come from backend validate JSON</span></div>
        </div>
      </div>
      <details class="rawjson"><summary>Show raw JSON</summary><pre>${U.esc(JSON.stringify(r, null, 2))}</pre></details>
    </div>
  `;
}

function prettyCompareHTML(r) {
  const file = r.file?.name || r.file || "—";
  const pctiles = r.percentiles || {};
  const div = r.divergence || {};
  const score = r.score || {};
  const base = r.baseline || {};
  const pL = U.pct(pctiles.length_pct ?? pctiles.length);
  const pGC = U.pct(pctiles.gc_pct ?? pctiles.gc);
  const pN = U.pct(pctiles.n_pct ?? pctiles.n);
  const js = U.pct(div.k3_js ?? div.js ?? div.k3);
  const overall = U.pct(score.overall_0_100 ?? score.overall ?? score.score);
  const clsOverall = overall != null && overall >= 80 ? "good" : overall != null && overall >= 55 ? "warn" : "bad";
  const weirdness = js == null ? null : U.clamp(js, 0, 1) * 100;
  const clsWeird = weirdness != null && weirdness <= 20 ? "good" : weirdness != null && weirdness <= 45 ? "warn" : "bad";

  return `
    <div class="pretty">
      <div class="title"><div class="h">Compare → training distribution</div><div class="sub">${U.esc(file)} • baseline n=${U.esc(base.n_files ?? base.n ?? "—")}</div></div>
      <div class="pretty-grid">
        <div class="box">
          <div class="badges">
            ${badge(clsOverall, "overall", overall == null ? "—" : overall.toFixed(1) + "/100")}
            ${badge(clsWeird, "k3 weirdness", weirdness == null ? "—" : weirdness.toFixed(1) + "%")}
          </div>
          <div class="row"><span class="k">k=3 JS</span><span class="v">${js == null ? "—" : js.toFixed(4)}</span></div>
          ${meterHtml(100 - (weirdness ?? 0), "more typical", "more unusual")}
        </div>
        <div class="box">
          <div class="row"><span class="k">length pct</span><span class="v">${pL == null ? "—" : pL.toFixed(1)}</span></div>
          ${pctBarHtml(pL ?? 0)}
          <div class="row" style="margin-top:10px"><span class="k">GC pct</span><span class="v">${pGC == null ? "—" : pGC.toFixed(1)}</span></div>
          ${pctBarHtml(pGC ?? 0)}
          <div class="row" style="margin-top:10px"><span class="k">N pct</span><span class="v">${pN == null ? "—" : pN.toFixed(1)}</span></div>
          ${pctBarHtml(pN ?? 0)}
        </div>
      </div>
      <details class="rawjson"><summary>Show raw JSON</summary><pre>${U.esc(JSON.stringify(r, null, 2))}</pre></details>
    </div>
  `;
}

export function loadGeneratedFromLS() {
  const j = Store.getJSON(LS.generated, []);
  return Array.isArray(j) ? j : [];
}

export function setGenerated(list) {
  state.generated = Array.isArray(list) ? list : [];
  Store.setJSON(LS.generated, state.generated);
  renderGenerated();
}

export function renderGenerated() {
  const host = U.$("#generatedList");
  if (!host) return;

  host.innerHTML = "";
  if (!state.generated.length) {
    host.appendChild(U.el("div", { class: "pad muted", text: "No generated sequences yet." }));
    return;
  }

  for (const g of state.generated) {
    const file = g.file || g.name || g.filename || "—";
    const kind = g.kind || g.mode || "—";
    const item = U.el(
      "div",
      { class: "item" },
      U.el("div", { class: "l" }, U.el("div", { class: "mono", text: file }), U.el("div", { class: "muted", text: kind })),
      U.el("div", { class: "r" },
        U.el("div", { class: "row", style: "padding:0;justify-content:flex-end;gap:8px" },
          U.el("button", { class: "btn ghost", text: "Map", onclick: () => doMap(file) }),
          U.el("button", { class: "btn ghost", text: "Validate", onclick: () => doValidate(file) }),
          U.el("button", { class: "btn ghost", text: "Compare", onclick: () => doCompare(file) })
        )
      )
    );
    host.appendChild(item);
  }
}

function normalizeGcInput(v) {
  if (v === "" || v == null) return null;
  const n = Number(v);
  if (!Number.isFinite(n)) return null;
  if (n > 1.0) return U.clamp(n / 100.0, 0, 1);
  return U.clamp(n, 0, 1);
}

export async function doGenerate() {
  const mode = (U.$("#genMode")?.value || "genome").trim();
  const length = Number((U.$("#genLen")?.value || "").trim());
  const gc_target = normalizeGcInput((U.$("#genGc")?.value || "").trim());
  const seed = (U.$("#genSeed")?.value || "").trim();

  const payload = {
    mode,
    kind: mode,
    length: Number.isFinite(length) ? length : undefined,
    length_bp: mode === "genome" ? (Number.isFinite(length) ? length : undefined) : undefined,
    length_aa: mode === "protein" ? (Number.isFinite(length) ? length : undefined) : undefined,
    gc_target: gc_target ?? undefined,
    seed: seed || undefined,
  };
  for (const k of Object.keys(payload)) if (payload[k] === undefined) delete payload[k];

  const btn = U.$("#btnGenerate");
  if (btn) { btn.disabled = true; btn.textContent = "Generating…"; }

  try {
    const tries = mode === "protein" ? ["/generate/protein", "/generate", "/generate/genome"] : ["/generate/genome", "/generate", "/generate/protein"];
    let out = null, used = null, lastErr = null;

    for (const p of tries) {
      try { const r = await api.post(p, payload); out = r.json ?? null; used = p; break; }
      catch (e) { lastErr = e; }
    }
    if (!out) throw lastErr || new Error("No generate endpoint worked");

    const file = out.file?.name || out.file || out.filename || out.name;
    Toasts.toast(file ? `Generated: ${file}` : `Generated (${used})`, file ? "good" : "warn", 2800);

    const entry = {
      file: file || `generated_${Date.now()}`,
      kind: out.kind || mode,
      created_utc: out.time_utc || out.created_utc || new Date().toISOString(),
      meta: out,
    };
    setGenerated([entry, ...state.generated].slice(0, 50));
  } catch (e) {
    Toasts.toast(`Generate failed: ${e.message}`, "bad", 5600);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = "Generate"; }
  }
}

export async function doMap(file) {
  const outEl = U.$("#mapOut");
  if (!outEl) return;
  outEl.classList.remove("hidden");
  outEl.innerHTML = `<div class="pretty"><div class="title"><div class="h">Map</div><div class="sub">loading…</div></div></div>`;

  const tries = [
    api.url(`/generated/${encodeURIComponent(file)}/map`),
    api.url(`/generated/${encodeURIComponent(file)}/map.pdf`),
    api.url(`/generated/${encodeURIComponent(file)}/map?format=pdf`),
  ];

  let chosen = null;
  for (const u of tries) {
    try { const res = await fetch(u, { method: "GET" }); if (res.ok) { chosen = u; break; } } catch {}
  }

  if (!chosen) {
    outEl.innerHTML = `<div class="pretty"><div class="title"><div class="h">Map</div><div class="sub">not available</div></div><div class="pad muted">No map endpoint found for <span class="mono">${U.esc(file)}</span>.</div></div>`;
    return Toasts.toast("Map endpoint not found", "warn", 4200);
  }

  outEl.innerHTML = `
    <div class="pretty">
      <div class="title"><div class="h">Map</div><div class="sub">${U.esc(file)}</div></div>
      <div class="pad">
        <div class="row" style="padding:0;justify-content:flex-end">
          <a class="btn ghost" href="${chosen}" target="_blank" rel="noopener">Open map</a>
        </div>
        <div style="height:520px;margin-top:10px;border:1px solid rgba(255,255,255,.10);border-radius:14px;overflow:hidden">
          <iframe title="map" src="${chosen}" style="width:100%;height:100%;border:0"></iframe>
        </div>
      </div>
    </div>
  `;
}

export async function doValidate(file) {
  const outEl = U.$("#validateOut");
  if (!outEl) return;
  outEl.classList.remove("hidden");
  outEl.innerHTML = `<div class="pretty"><div class="title"><div class="h">Validate</div><div class="sub">working…</div></div></div>`;
  try {
    const r = await api.post(`/generated/${encodeURIComponent(file)}/validate`, {});
    outEl.innerHTML = prettyValidateHTML(r.json ?? {});
  } catch (e) {
    outEl.innerHTML = `<div class="pretty"><div class="title"><div class="h">Validate</div><div class="sub">error</div></div><div class="pad mono">${U.esc(e.message)}</div></div>`;
    Toasts.toast(`Validate failed: ${e.message}`, "bad", 5200);
  }
}

export async function doCompare(file) {
  const outEl = U.$("#compareOut");
  if (!outEl) return;
  outEl.classList.remove("hidden");
  outEl.innerHTML = `<div class="pretty"><div class="title"><div class="h">Compare</div><div class="sub">working…</div></div></div>`;
  try {
    const r = await api.post(`/generated/${encodeURIComponent(file)}/compare`, {});
    outEl.innerHTML = prettyCompareHTML(r.json ?? {});
  } catch (e) {
    outEl.innerHTML = `<div class="pretty"><div class="title"><div class="h">Compare</div><div class="sub">error</div></div><div class="pad mono">${U.esc(e.message)}</div></div>`;
    Toasts.toast(`Compare failed: ${e.message}`, "bad", 5200);
  }
}
