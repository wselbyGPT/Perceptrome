"use strict";

/**
 * Perceptrome Dashboard (vanilla JS)
 * - Works with nginx serving / (static) and proxying /api/* to FastAPI (127.0.0.1:9000)
 * - Robust: if expected DOM nodes aren't present, it builds a complete UI inside #app or <body>.
 *
 * Expected API endpoints (best-effort; some are optional):
 *   GET  /api/health
 *   GET  /api/catalogs
 *   GET  /api/cache
 *   GET  /api/genome/{acc}/summary                 (requested)
 *
 * Optional generation endpoints (tries several):
 *   POST /api/generate
 *   POST /api/generate/genome
 *   POST /api/generate/protein
 *
 * Optional per-file endpoints:
 *   POST /api/generated/{file}/validate
 *   POST /api/generated/{file}/compare
 *   GET  /api/generated/{file}/map
 *   GET  /api/generated/{file}/map.pdf
 *
 * Optional catalog listing endpoints (tries several):
 *   GET /api/catalog/{catalog}?limit=...&offset=...
 *   GET /api/catalogs/{catalog}?limit=...&offset=...
 *   GET /api/catalog/{catalog}/accessions?limit=...&offset=...
 *   GET /api/catalog/{catalog}/preview?limit=...&offset=...
 */

const API_PREFIX = "/api";
const LS = {
  generated: "perceptrome.generated.v1",
  lastCatalog: "perceptrome.lastCatalog.v1",
  lastAcc: "perceptrome.lastAcc.v1",
};

function $(sel, root=document){ return root.querySelector(sel); }
function $all(sel, root=document){ return Array.from(root.querySelectorAll(sel)); }

function el(tag, attrs={}, ...kids){
  const n = document.createElement(tag);
  for (const [k,v] of Object.entries(attrs||{})){
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on") && typeof v === "function") n.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null) n.setAttribute(k, String(v));
  }
  for (const kid of kids){
    if (kid === null || kid === undefined) continue;
    if (typeof kid === "string") n.appendChild(document.createTextNode(kid));
    else n.appendChild(kid);
  }
  return n;
}

function esc(s){
  return String(s ?? "")
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;")
    .replaceAll("'","&#39;");
}

function fmtBytes(b){
  b = Number(b||0);
  const u = ["B","KB","MB","GB","TB"];
  let i = 0;
  while (b >= 1024 && i < u.length-1){ b/=1024; i++; }
  return `${b.toFixed(i===0?0:1)} ${u[i]}`;
}

function fmtIso(iso){
  if(!iso) return "—";
  try{
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    return d.toLocaleString();
  }catch{ return String(iso); }
}

function pct(x){
  if (x === null || x === undefined) return null;
  const n = Number(x);
  return Number.isFinite(n) ? n : null;
}

function clamp(v,a,b){ return Math.max(a, Math.min(b, v)); }

function sleep(ms){ return new Promise(r=>setTimeout(r, ms)); }

function apiUrl(path){
  if (!path) path = "/";
  if (!path.startsWith("/")) path = "/" + path;
  if (path.startsWith(API_PREFIX + "/")) return path;
  return API_PREFIX + path;
}

async function apiFetch(path, opts={}){
  const url = apiUrl(path);
  const res = await fetch(url, opts);
  const ctype = (res.headers.get("content-type")||"").toLowerCase();
  let bodyText = null;
  let bodyJson = null;

  if (!res.ok){
    try { bodyText = await res.text(); } catch {}
    const msg = bodyText ? bodyText.slice(0, 600) : `${res.status} ${res.statusText}`;
    throw new Error(`${res.status} ${res.statusText}: ${msg}`);
  }

  if (ctype.includes("application/json")){
    bodyJson = await res.json();
    return { res, json: bodyJson, text: null, ctype };
  } else {
    bodyText = await res.text();
    // Some endpoints may return JSON without proper header; try parse.
    const trimmed = bodyText.trim();
    if (trimmed.startsWith("{") || trimmed.startsWith("[")){
      try { bodyJson = JSON.parse(trimmed); return { res, json: bodyJson, text: bodyText, ctype }; } catch {}
    }
    return { res, json: null, text: bodyText, ctype };
  }
}

async function apiGet(path){
  return apiFetch(path, { method: "GET" });
}

async function apiPost(path, obj){
  return apiFetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(obj ?? {})
  });
}

/* ---------- toast ---------- */

function ensureToast(){
  let t = $("#toast");
  if (!t){
    t = el("div", { id:"toast", class:"toast", "aria-live":"polite" });
    document.body.appendChild(t);
  }
  return t;
}

function toast(msg, kind="info", ms=3200){
  const host = ensureToast();
  const item = el("div", { class:`toast-item ${kind}`, html: esc(msg) });
  host.appendChild(item);
  setTimeout(()=>{ item.classList.add("out"); }, ms);
  setTimeout(()=>{ item.remove(); }, ms+500);
}

/* ---------- UI builders / finders ---------- */

function ensureUI(){
  // If index.html already has a full UI, we’ll just bind to it.
  // If not, we generate a complete layout inside #app or body.
  const root = $("#app") || document.body;

  // Heuristic: if #catalogList exists, assume page already scaffolded.
  if ($("#catalogList") && $("#accessionList") && $("#inspector")) return;

  // Build scaffold
  root.innerHTML = "";
  root.appendChild(el("div", { class:"wrap" },
    el("header", { class:"top" },
      el("div", { class:"brand" },
        el("div", { class:"dot" }),
        el("div", { class:"title", html:`<b>Perceptrome</b> <span class="muted">Genome Dashboard</span>` })
      ),
      el("div", { class:"statusline" },
        el("span", { class:"pill", id:"pillHealth", html:"API: <b>…</b>" }),
        el("span", { class:"pill", id:"pillRepo", html:"repo: <b>…</b>" }),
        el("span", { class:"pill", id:"pillTime", html:"time: <b>…</b>" }),
      )
    ),
    el("div", { class:"grid" },
      el("aside", { class:"side" },
        el("section", { class:"card" },
          el("div", { class:"card-h", html:`<b>Catalogs</b><span class="muted" id="catalogCount"></span>` }),
          el("div", { class:"list", id:"catalogList" }),
        ),
        el("section", { class:"card" },
          el("div", { class:"card-h", html:`<b>Cache</b><span class="muted" id="cacheSummary"></span>` }),
          el("div", { class:"list", id:"cacheList" }),
        ),
      ),
      el("main", { class:"main" },
        el("section", { class:"card" },
          el("div", { class:"card-h", html:`<b>Accessions</b><span class="muted" id="accMeta"></span>` }),
          el("div", { class:"row" },
            el("input", { id:"accessionSearch", class:"input", placeholder:"search accessions (filter)…" }),
            el("button", { id:"btnReloadAcc", class:"btn", text:"Reload" }),
            el("button", { id:"btnLoadMoreAcc", class:"btn ghost", text:"Load more", title:"if API supports offset/pagination" }),
          ),
          el("div", { class:"list tall", id:"accessionList" })
        ),
        el("section", { class:"card" },
          el("div", { class:"card-h", html:`<b>Generate</b><span class="muted">→ Map / Validate / Compare</span>` }),
          el("div", { class:"row wraprow" },
            el("select", { id:"genMode", class:"input" },
              el("option", { value:"genome", text:"genome" }),
              el("option", { value:"protein", text:"protein" }),
            ),
            el("input", { id:"genLen", class:"input", placeholder:"length (bp or aa)", value:"20000" }),
            el("input", { id:"genGc", class:"input", placeholder:"GC target (0-1 or %)", value:"0.50" }),
            el("input", { id:"genSeed", class:"input", placeholder:"seed (optional)" }),
            el("button", { id:"btnGenerate", class:"btn", text:"Generate" }),
          ),
          el("div", { class:"list", id:"generatedList" }),
          el("div", { class:"row wraprow" },
            el("button", { id:"btnClearGenerated", class:"btn ghost", text:"Clear list" }),
            el("span", { class:"muted", id:"genHint", text:"Tip: generated files persist in ./generated; this list is localStorage." }),
          ),
        )
      ),
      el("aside", { class:"inspect" },
        el("section", { class:"card", id:"inspector" },
          el("div", { class:"card-h", html:`<b>Genome Inspector</b><span class="muted" id="inspectorMeta"></span>` }),
          el("div", { id:"inspectorBody", class:"pad", html:`<div class="muted">Click an accession to view summary (length, GC%, N%, features).</div>` })
        ),
        el("section", { class:"card" },
          el("div", { class:"card-h", html:`<b>Outputs</b><span class="muted">map / validate / compare</span>` }),
          el("div", { id:"mapOut", class:"pad hidden" }),
          el("div", { id:"validateOut", class:"pad hidden" }),
          el("div", { id:"compareOut", class:"pad hidden" }),
        )
      )
    )
  ));

  // Minimal inline styles if your CSS doesn't include layout (keeps it usable).
  if (!$("#__perceptrome_inline_style")){
    document.head.appendChild(el("style", { id:"__perceptrome_inline_style", html: `
      .wrap{max-width:1400px;margin:0 auto;padding:18px}
      .top{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:14px}
      .brand{display:flex;align-items:center;gap:10px}
      .dot{width:12px;height:12px;border-radius:999px;background:rgba(120,190,255,.75);box-shadow:0 0 24px rgba(120,190,255,.35)}
      .title{font-size:16px}
      .muted{color:rgba(255,255,255,.60);font-size:12px;margin-left:8px}
      .statusline{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}
      .pill{border:1px solid rgba(255,255,255,.12);background:rgba(0,0,0,.20);padding:6px 10px;border-radius:999px;font-size:12px}
      .grid{display:grid;grid-template-columns:320px 1fr 420px;gap:12px}
      @media (max-width:1200px){.grid{grid-template-columns:1fr}.side,.inspect{order:2}}
      .card{border:1px solid rgba(255,255,255,.10);background:rgba(0,0,0,.18);border-radius:18px;overflow:hidden}
      .card-h{display:flex;justify-content:space-between;align-items:baseline;gap:10px;padding:10px 12px;border-bottom:1px solid rgba(255,255,255,.08)}
      .list{display:flex;flex-direction:column}
      .list.tall{max-height:360px;overflow:auto}
      .item{display:flex;justify-content:space-between;gap:10px;padding:9px 12px;border-bottom:1px solid rgba(255,255,255,.06);cursor:pointer}
      .item:hover{background:rgba(255,255,255,.04)}
      .item .l{display:flex;flex-direction:column;gap:2px;min-width:0}
      .item .r{display:flex;flex-direction:column;gap:2px;text-align:right}
      .mono{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace}
      .row{display:flex;gap:10px;align-items:center;padding:10px 12px}
      .wraprow{flex-wrap:wrap}
      .input{flex:1;min-width:180px;padding:9px 10px;border-radius:12px;border:1px solid rgba(255,255,255,.12);background:rgba(0,0,0,.22);color:rgba(255,255,255,.92)}
      .btn{padding:9px 12px;border-radius:12px;border:1px solid rgba(255,255,255,.14);background:rgba(120,190,255,.22);color:rgba(255,255,255,.92);cursor:pointer}
      .btn:hover{filter:brightness(1.05)}
      .btn.ghost{background:rgba(0,0,0,.18)}
      .pad{padding:10px 12px}
      .hidden{display:none !important}
      .toast{position:fixed;right:14px;bottom:14px;display:flex;flex-direction:column;gap:10px;z-index:9999}
      .toast-item{border:1px solid rgba(255,255,255,.14);background:rgba(0,0,0,.55);backdrop-filter:blur(8px);
        padding:10px 12px;border-radius:14px;max-width:420px}
      .toast-item.out{opacity:0;transform:translateY(10px);transition:all .35s ease}
      .toast-item.good{border-color:rgba(120,255,170,.28)}
      .toast-item.warn{border-color:rgba(255,200,120,.28)}
      .toast-item.bad{border-color:rgba(255,77,109,.35)}
    `}));
  }
}

/* ---------- Pretty blocks (Validate/Compare) ---------- */

function meterHtml(value0to100, leftLabel, rightLabel){
  const w = clamp(value0to100 ?? 0, 0, 100);
  return `
    <div class="meter"><div class="fill" style="width:${w}%"></div></div>
    <div class="meter-label"><span>${esc(leftLabel)}</span><span>${esc(rightLabel)}</span></div>
  `;
}

function pctBarHtml(pct0to100){
  const p = clamp(pct0to100 ?? 0, 0, 100);
  return `
    <div class="pctbar">
      <div class="zone"></div>
      <div class="mark" style="left: calc(${p}% - 1px)"></div>
    </div>
    <div class="meter-label"><span>0</span><span>25</span><span>50</span><span>75</span><span>100</span></div>
  `;
}

function badge(cls, k, v){
  return `<div class="bx ${cls}"><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span></div>`;
}

function prettyValidateHTML(r){
  const kind = r.kind || "—";
  const file = r.file?.name || r.file || "—";
  const len = Number(r.length ?? 0).toLocaleString();

  if(kind === "protein"){
    const x = pct(r.x_pct);
    const mr = r.max_run || {};
    const run = mr.run ?? 0;
    const base = mr.base ?? "—";
    const clsX = (x != null && x <= 2) ? "good" : (x != null && x <= 10 ? "warn" : "bad");
    const clsRun = (run <= 10) ? "good" : (run <= 25 ? "warn" : "bad");
    const top = (r.aa_counts_top || []).map(([aa,c]) => `${aa}:${c}`).join("  ");

    return `
      <div class="pretty">
        <div class="title">
          <div class="h">Validate</div>
          <div class="sub">${esc(file)} • protein</div>
        </div>

        <div class="grid">
          <div class="box">
            <div class="row"><span class="k">length</span><span class="v">${esc(len)}</span></div>
            <div class="row"><span class="k">X% (unknown/invalid)</span><span class="v">${x == null ? "—" : x.toFixed(2) + "%"}</span></div>
            ${meterHtml(100 - clamp(x ?? 0, 0, 100), "more clean", "more X")}
            <div class="badges">
              ${badge(clsX, "X%", x == null ? "—" : x.toFixed(2) + "%")}
              ${badge(clsRun, "max-run", `${base}×${run}`)}
            </div>
          </div>

          <div class="box">
            <div class="row"><span class="k">homopolymer max-run</span><span class="v">${esc(`${base}×${run}`)}</span></div>
            <div class="row"><span class="k">AA counts (top)</span><span class="v">${esc(top || "—")}</span></div>
          </div>
        </div>

        <details class="rawjson">
          <summary>Show raw JSON</summary>
          <pre>${esc(JSON.stringify(r, null, 2))}</pre>
        </details>
      </div>
    `;
  }

  // genome
  const gc = pct(r.gc_pct);
  const np = pct(r.n_pct);
  const inv = pct(r.invalid_pct);
  const mr = r.max_run || {};
  const run = mr.run ?? 0;
  const base = mr.base ?? "—";
  const orf = r.orfs_forward || r.orfs || {};
  const orfCount = orf.count ?? 0;
  const longest = orf.longest_aa ?? 0;
  const topOrfs = (orf.top || []).slice(0, 8);

  const clsN = (np != null && np <= 1) ? "good" : (np != null && np <= 5 ? "warn" : "bad");
  const clsInv = (inv != null && inv <= 0.1) ? "good" : (inv != null && inv <= 1 ? "warn" : "bad");
  const clsRun = (run <= 12) ? "good" : (run <= 30 ? "warn" : "bad");
  const clsOrf = (orfCount >= 2 && longest >= 200) ? "good" : (orfCount >= 1 ? "warn" : "bad");

  const orfRows = topOrfs.length ? topOrfs.map((o, i) => `
    <tr>
      <td>${i+1}</td>
      <td>${esc(o.frame ?? o.f ?? "—")}</td>
      <td>${esc(o.aa_len ?? o.aa ?? "—")}</td>
      <td>${esc(o.start ?? "—")}–${esc(o.end ?? "—")}</td>
    </tr>
  `).join("") : `<tr><td colspan="4">No ORFs above threshold.</td></tr>`;

  return `
    <div class="pretty">
      <div class="title">
        <div class="h">Validate</div>
        <div class="sub">${esc(file)} • genome</div>
      </div>

      <div class="grid">
        <div class="box">
          <div class="row"><span class="k">length</span><span class="v">${esc(len)} bp</span></div>
          <div class="row"><span class="k">GC%</span><span class="v">${gc == null ? "—" : gc.toFixed(2) + "%"}</span></div>
          <div class="row"><span class="k">N%</span><span class="v">${np == null ? "—" : np.toFixed(2) + "%"}</span></div>
          <div class="row"><span class="k">invalid%</span><span class="v">${inv == null ? "—" : inv.toFixed(3) + "%"}</span></div>

          <div class="badges">
            ${badge(clsN, "N%", np == null ? "—" : np.toFixed(2) + "%")}
            ${badge(clsInv, "invalid", inv == null ? "—" : inv.toFixed(3) + "%")}
            ${badge(clsRun, "max-run", `${base}×${run}`)}
            ${badge(clsOrf, "ORFs", `${orfCount} (longest ${longest} aa)`)}
          </div>
        </div>

        <div class="box">
          <div class="row"><span class="k">ORFs</span><span class="v">${esc(orfCount)} • longest ${esc(longest)} aa</span></div>
          <table class="tbl">
            <thead><tr><th>#</th><th>frame</th><th>aa</th><th>pos</th></tr></thead>
            <tbody>${orfRows}</tbody>
          </table>
        </div>
      </div>

      <details class="rawjson">
        <summary>Show raw JSON</summary>
        <pre>${esc(JSON.stringify(r, null, 2))}</pre>
      </details>
    </div>
  `;
}

function prettyCompareHTML(r){
  const file = r.file?.name || r.file || "—";
  const seq = r.sequence || {};
  const pctiles = r.percentiles || {};
  const div = r.divergence || {};
  const score = r.score || {};
  const base = r.baseline || {};

  const L = seq.length ?? seq.len ?? 0;
  const gc = pct(seq.gc_pct ?? seq.gc);
  const np = pct(seq.n_pct ?? seq.n);

  const pL = pct(pctiles.length_pct ?? pctiles.length);
  const pGC = pct(pctiles.gc_pct ?? pctiles.gc);
  const pN = pct(pctiles.n_pct ?? pctiles.n);

  const js = pct(div.k3_js ?? div.js ?? div.k3);
  const overall = pct(score.overall_0_100 ?? score.overall ?? score.score);

  const clsOverall = (overall != null && overall >= 80) ? "good" : (overall != null && overall >= 55 ? "warn" : "bad");
  const weirdness = js == null ? null : clamp(js, 0, 1) * 100; // 0% = typical, 100% = unusual
  const clsWeird = (weirdness != null && weirdness <= 20) ? "good" : (weirdness != null && weirdness <= 45 ? "warn" : "bad");

  return `
    <div class="pretty">
      <div class="title">
        <div class="h">Compare → training distribution</div>
        <div class="sub">${esc(file)} • baseline n=${esc(base.n_files ?? base.n ?? "—")}</div>
      </div>

      <div class="grid">
        <div class="box">
          <div class="row"><span class="k">length</span><span class="v">${Number(L).toLocaleString()} bp</span></div>
          <div class="row"><span class="k">GC%</span><span class="v">${gc == null ? "—" : gc.toFixed(2) + "%"}</span></div>
          <div class="row"><span class="k">N%</span><span class="v">${np == null ? "—" : np.toFixed(2) + "%"}</span></div>

          <div class="badges">
            ${badge(clsOverall, "overall", overall == null ? "—" : overall.toFixed(1) + "/100")}
            ${badge(clsWeird, "k3 weirdness", weirdness == null ? "—" : weirdness.toFixed(1) + "%")}
          </div>

          <div class="row" style="margin-top:10px"><span class="k">k=3 JS divergence</span><span class="v">${js == null ? "—" : js.toFixed(4)}</span></div>
          ${meterHtml(100 - (weirdness ?? 0), "more typical", "more unusual")}
          <div class="sub" style="margin-top:10px">built ${esc((base.built_utc || base.built || "").replace("T"," ").replace("+00:00","Z"))}</div>
        </div>

        <div class="box">
          <div class="row"><span class="k">Percentiles (0–100)</span><span class="v">marker = your value</span></div>

          <div class="row"><span class="k">length pct</span><span class="v">${pL == null ? "—" : pL.toFixed(1)}</span></div>
          ${pctBarHtml(pL ?? 0)}

          <div class="row" style="margin-top:10px"><span class="k">GC pct</span><span class="v">${pGC == null ? "—" : pGC.toFixed(1)}</span></div>
          ${pctBarHtml(pGC ?? 0)}

          <div class="row" style="margin-top:10px"><span class="k">N pct</span><span class="v">${pN == null ? "—" : pN.toFixed(1)}</span></div>
          ${pctBarHtml(pN ?? 0)}
        </div>
      </div>

      <details class="rawjson">
        <summary>Show raw JSON</summary>
        <pre>${esc(JSON.stringify(r, null, 2))}</pre>
      </details>
    </div>
  `;
}

/* ---------- Genome Inspector ---------- */

function prettyGenomeSummaryHTML(s){
  // Accept flexible keys
  const acc = s.acc || s.accession || s.id || "—";
  const length = Number(s.length_bp ?? s.length ?? 0);
  const gc = pct(s.gc_pct ?? s.gc);
  const np = pct(s.n_pct ?? s.n);
  const inv = pct(s.invalid_pct ?? s.invalid);

  const feats = s.features || s.feature_counts || {};
  const cds = feats.cds ?? feats.CDS ?? feats.cds_count ?? s.cds_count;
  const gene = feats.gene ?? feats.genes ?? feats.gene_count ?? s.gene_count;
  const trna = feats.trna ?? feats.tRNA ?? feats.trna_count ?? s.trna_count;
  const rrna = feats.rrna ?? feats.rRNA ?? feats.rrna_count ?? s.rrna_count;

  const clsN = (np != null && np <= 1) ? "good" : (np != null && np <= 5 ? "warn" : "bad");

  return `
    <div class="pretty">
      <div class="title">
        <div class="h">Genome Inspector</div>
        <div class="sub">${esc(acc)}</div>
      </div>

      <div class="grid">
        <div class="box">
          <div class="row"><span class="k">length</span><span class="v">${length ? length.toLocaleString() + " bp" : "—"}</span></div>
          <div class="row"><span class="k">GC%</span><span class="v">${gc == null ? "—" : gc.toFixed(2) + "%"}</span></div>
          <div class="row"><span class="k">N%</span><span class="v">${np == null ? "—" : np.toFixed(2) + "%"}</span></div>
          <div class="row"><span class="k">invalid%</span><span class="v">${inv == null ? "—" : inv.toFixed(3) + "%"}</span></div>

          <div class="badges">
            ${badge(clsN, "N%", np == null ? "—" : np.toFixed(2) + "%")}
            ${badge("good", "GC", gc == null ? "—" : gc.toFixed(2) + "%")}
            ${badge("warn", "len", length ? length.toLocaleString() : "—")}
          </div>
        </div>

        <div class="box">
          <div class="row"><span class="k">features</span><span class="v">counts</span></div>
          <div class="badges">
            ${badge("good", "CDS", cds ?? "—")}
            ${badge("good", "gene", gene ?? "—")}
            ${badge("good", "tRNA", trna ?? "—")}
            ${badge("good", "rRNA", rrna ?? "—")}
          </div>
        </div>
      </div>

      <details class="rawjson">
        <summary>Show raw JSON</summary>
        <pre>${esc(JSON.stringify(s, null, 2))}</pre>
      </details>
    </div>
  `;
}

async function loadGenomeSummary(acc){
  const body = $("#inspectorBody");
  const meta = $("#inspectorMeta");
  if (meta) meta.textContent = acc;
  if (body) body.innerHTML = `<div class="muted">Loading summary for <span class="mono">${esc(acc)}</span>…</div>`;

  try{
    const { json } = await apiGet(`/genome/${encodeURIComponent(acc)}/summary`);
    localStorage.setItem(LS.lastAcc, acc);
    if (body) body.innerHTML = prettyGenomeSummaryHTML(json || {});
  }catch(e){
    if (body) body.innerHTML = `<div class="pretty"><div class="title"><div class="h">Inspector</div><div class="sub">error</div></div><div class="pad mono">${esc(e.message)}</div></div>`;
    toast(`Inspector failed: ${e.message}`, "bad", 4500);
  }
}

/* ---------- Catalogs / Accessions ---------- */

const state = {
  catalogs: [],
  cache: null,
  currentCatalog: null,
  accAll: [],
  accFiltered: [],
  accOffset: 0,
  accLimit: 2000,
  generated: [],
};

function setGenerated(list){
  state.generated = Array.isArray(list) ? list : [];
  try { localStorage.setItem(LS.generated, JSON.stringify(state.generated)); } catch {}
  renderGenerated();
}

function loadGeneratedFromLS(){
  try{
    const s = localStorage.getItem(LS.generated);
    if (!s) return [];
    const j = JSON.parse(s);
    return Array.isArray(j) ? j : [];
  }catch{ return []; }
}

function renderCatalogs(){
  const host = $("#catalogList");
  const count = $("#catalogCount");
  if (!host) return;

  host.innerHTML = "";
  if (count) count.textContent = state.catalogs.length ? `(${state.catalogs.length})` : "";

  for (const it of state.catalogs){
    const name = it.catalog || it.name || "—";
    const bytes = fmtBytes(it.bytes ?? 0);
    const mtime = fmtIso(it.mtime);
    const item = el("div", { class:"item", role:"button", tabindex:"0" },
      el("div", { class:"l" },
        el("div", { class:"mono", text: name }),
        el("div", { class:"muted", text: it.name || "" }),
      ),
      el("div", { class:"r" },
        el("div", { class:"muted", text: bytes }),
        el("div", { class:"muted", text: mtime }),
      ),
    );
    item.addEventListener("click", ()=>selectCatalog(name));
    item.addEventListener("keydown", (e)=>{ if(e.key==="Enter"||e.key===" "){ e.preventDefault(); selectCatalog(name);} });
    host.appendChild(item);
  }
}

function renderCache(){
  const host = $("#cacheList");
  const sum = $("#cacheSummary");
  if (!host) return;
  host.innerHTML = "";

  const c = state.cache;
  if (!c){ if(sum) sum.textContent=""; return; }

  const parts = [];
  for (const key of ["genbank","fasta","encoded"]){
    const b = c[key];
    if (!b) continue;
    parts.push(`${key}:${b.count ?? 0}`);
  }
  if (sum) sum.textContent = parts.length ? `(${parts.join(" • ")})` : "";

  for (const key of ["genbank","fasta","encoded"]){
    const b = c[key];
    if (!b) continue;
    const item = el("div", { class:"item" },
      el("div", { class:"l" },
        el("div", { class:"mono", text: key }),
        el("div", { class:"muted", text: b.path || "" }),
      ),
      el("div", { class:"r" },
        el("div", { class:"muted", text: (b.count ?? 0).toLocaleString() }),
        el("div", { class:"muted", text: b.exists ? "ok" : "missing" }),
      ),
    );
    host.appendChild(item);
  }
}

function renderAccessions(){
  const host = $("#accessionList");
  const meta = $("#accMeta");
  if (!host) return;

  const total = state.accAll.length;
  const shown = state.accFiltered.length;
  if (meta) meta.textContent = state.currentCatalog ? `${state.currentCatalog} • ${shown.toLocaleString()} shown / ${total.toLocaleString()} loaded` : "";

  host.innerHTML = "";

  const maxShow = Math.min(shown, 2500); // keep DOM light
  for (let i=0; i<maxShow; i++){
    const acc = state.accFiltered[i];
    const item = el("div", { class:"item", role:"button", tabindex:"0" },
      el("div", { class:"l" },
        el("div", { class:"mono", text: acc }),
        el("div", { class:"muted", text: "click → inspector" }),
      ),
      el("div", { class:"r" },
        el("div", { class:"muted", text: "genome" }),
        el("div", { class:"muted", text: "summary" }),
      ),
    );
    item.addEventListener("click", ()=>loadGenomeSummary(acc));
    item.addEventListener("keydown", (e)=>{ if(e.key==="Enter"||e.key===" "){ e.preventDefault(); loadGenomeSummary(acc);} });
    host.appendChild(item);
  }

  if (shown > maxShow){
    host.appendChild(el("div", { class:"pad muted", html: `Showing first <b>${maxShow.toLocaleString()}</b> of <b>${shown.toLocaleString()}</b> (filter to narrow).` }));
  }
}

function applyAccFilter(){
  const q = ($("#accessionSearch")?.value || "").trim().toLowerCase();
  if (!q){
    state.accFiltered = state.accAll.slice();
  } else {
    state.accFiltered = state.accAll.filter(a => a.toLowerCase().includes(q));
  }
  renderAccessions();
}

function parseAccessionsFromResponse(payload){
  // payload may be:
  // - JSON: { items:[...strings or objs...], accessions:[...], lines:[...] }
  // - JSON: [...strings]
  // - text: newline-separated
  if (!payload) return [];

  if (typeof payload === "string"){
    return payload.split(/\r?\n/).map(s=>s.trim()).filter(Boolean);
  }

  if (Array.isArray(payload)){
    return payload.map(x => (typeof x === "string" ? x : (x.acc || x.accession || x.id || ""))).filter(Boolean);
  }

  const arr =
    payload.accessions ||
    payload.items ||
    payload.lines ||
    payload.data ||
    null;

  if (Array.isArray(arr)){
    return arr.map(x => (typeof x === "string" ? x : (x.acc || x.accession || x.id || ""))).filter(Boolean);
  }

  return [];
}

async function fetchCatalogAccessions(catalog, limit=2000, offset=0){
  // try several possible endpoints
  const tries = [
    `/catalog/${encodeURIComponent(catalog)}?limit=${limit}&offset=${offset}`,
    `/catalogs/${encodeURIComponent(catalog)}?limit=${limit}&offset=${offset}`,
    `/catalog/${encodeURIComponent(catalog)}/accessions?limit=${limit}&offset=${offset}`,
    `/catalog/${encodeURIComponent(catalog)}/preview?limit=${limit}&offset=${offset}`,
    `/catalog/${encodeURIComponent(catalog)}`,
    `/catalogs/${encodeURIComponent(catalog)}`,
  ];

  let lastErr = null;
  for (const path of tries){
    try{
      const out = await apiGet(path);
      if (out.json !== null){
        const list = parseAccessionsFromResponse(out.json);
        return { list, raw: out.json, pathUsed: path };
      } else if (out.text !== null){
        const list = parseAccessionsFromResponse(out.text);
        return { list, raw: out.text, pathUsed: path };
      }
    }catch(e){
      lastErr = e;
      // continue
    }
  }
  throw lastErr || new Error("No catalog endpoint worked");
}

async function selectCatalog(catalog){
  state.currentCatalog = catalog;
  state.accAll = [];
  state.accFiltered = [];
  state.accOffset = 0;
  localStorage.setItem(LS.lastCatalog, catalog);

  toast(`Loading catalog: ${catalog}`, "info", 1800);

  try{
    const { list, pathUsed } = await fetchCatalogAccessions(catalog, state.accLimit, state.accOffset);
    state.accAll = list;
    state.accOffset += list.length;
    applyAccFilter();
    toast(`Loaded ${list.length.toLocaleString()} accessions (${pathUsed})`, "good", 2600);
  }catch(e){
    state.accAll = [];
    state.accFiltered = [];
    renderAccessions();
    toast(`Catalog load failed: ${e.message}`, "bad", 5200);
  }
}

async function loadMoreAccessions(){
  if (!state.currentCatalog) return toast("Pick a catalog first", "warn", 2600);
  try{
    const { list } = await fetchCatalogAccessions(state.currentCatalog, state.accLimit, state.accOffset);
    if (!list.length) return toast("No more returned (or API doesn’t paginate)", "warn", 3200);
    state.accAll.push(...list);
    state.accOffset += list.length;
    applyAccFilter();
    toast(`+${list.length.toLocaleString()} loaded`, "good", 2400);
  }catch(e){
    toast(`Load more failed: ${e.message}`, "bad", 5200);
  }
}

/* ---------- Generate / file actions ---------- */

function renderGenerated(){
  const host = $("#generatedList");
  if (!host) return;
  host.innerHTML = "";

  if (!state.generated.length){
    host.appendChild(el("div", { class:"pad muted", text:"No generated sequences yet." }));
    return;
  }

  for (const g of state.generated){
    const file = g.file || g.name || g.filename || "—";
    const kind = g.kind || g.mode || "—";
    const when = g.created_utc ? fmtIso(g.created_utc) : (g.time ? fmtIso(g.time) : "");
    const item = el("div", { class:"item" },
      el("div", { class:"l" },
        el("div", { class:"mono", text: file }),
        el("div", { class:"muted", text: `${kind}${when ? " • " + when : ""}` }),
      ),
      el("div", { class:"r" },
        el("div", { class:"row", style:"padding:0;justify-content:flex-end;gap:8px" },
          el("button", { class:"btn ghost", text:"Map", onclick:()=>doMap(file) }),
          el("button", { class:"btn ghost", text:"Validate", onclick:()=>doValidate(file, $("#validateOut")) }),
          el("button", { class:"btn ghost", text:"Compare", onclick:()=>doCompare(file, $("#compareOut")) }),
        ),
      ),
    );
    host.appendChild(item);
  }
}

function normalizeGcInput(v){
  if (v === "" || v == null) return null;
  const n = Number(v);
  if (!Number.isFinite(n)) return null;
  // allow 50 meaning 50%
  if (n > 1.0) return clamp(n/100.0, 0, 1);
  return clamp(n, 0, 1);
}

async function doGenerate(){
  const mode = ($("#genMode")?.value || "genome").trim();
  const lenRaw = ($("#genLen")?.value || "").trim();
  const gcRaw = ($("#genGc")?.value || "").trim();
  const seed = ($("#genSeed")?.value || "").trim();

  const length = Number(lenRaw);
  const gc_target = normalizeGcInput(gcRaw);

  const payload = {
    mode,
    kind: mode,
    length: Number.isFinite(length) ? length : undefined,
    length_bp: mode==="genome" ? (Number.isFinite(length) ? length : undefined) : undefined,
    length_aa: mode==="protein" ? (Number.isFinite(length) ? length : undefined) : undefined,
    gc_target: gc_target ?? undefined,
    seed: seed || undefined,
  };

  const btn = $("#btnGenerate");
  if (btn){ btn.disabled = true; btn.textContent = "Generating…"; }

  try{
    // try endpoints in order
    const tries = mode==="protein"
      ? ["/generate/protein", "/generate", "/generate/genome"]
      : ["/generate/genome", "/generate", "/generate/protein"];

    let out = null;
    let used = null;
    let lastErr = null;

    for (const p of tries){
      try{
        const r = await apiPost(p, payload);
        out = r.json ?? null;
        used = p;
        break;
      }catch(e){
        lastErr = e;
      }
    }
    if (!out) throw lastErr || new Error("No generate endpoint worked");

    const file = out.file?.name || out.file || out.filename || out.name;
    if (!file) toast(`Generated (endpoint ${used}) but no file name returned`, "warn", 5200);
    else toast(`Generated: ${file}`, "good", 2800);

    const entry = {
      file: file || `generated_${Date.now()}`,
      kind: out.kind || mode,
      created_utc: out.time_utc || out.created_utc || new Date().toISOString(),
      meta: out,
    };

    setGenerated([entry, ...state.generated].slice(0, 50));
    // auto-run compare/validate if you want later; for now just add.
  }catch(e){
    toast(`Generate failed: ${e.message}`, "bad", 5600);
  }finally{
    if (btn){ btn.disabled = false; btn.textContent = "Generate"; }
  }
}

async function doMap(file){
  const outEl = $("#mapOut");
  if (!outEl) return;

  outEl.classList.remove("hidden");
  outEl.innerHTML = `<div class="pretty"><div class="title"><div class="h">Map</div><div class="sub">loading…</div></div></div>`;

  // Try to embed a PDF if endpoint exists; else open new tab.
  const tries = [
    apiUrl(`/generated/${encodeURIComponent(file)}/map`),
    apiUrl(`/generated/${encodeURIComponent(file)}/map.pdf`),
    apiUrl(`/generated/${encodeURIComponent(file)}/map?format=pdf`),
  ];

  // Try HEAD-ish by fetching small range
  let chosen = null;
  for (const u of tries){
    try{
      const res = await fetch(u, { method:"GET" });
      if (res.ok){
        chosen = u;
        break;
      }
    }catch{}
  }

  if (!chosen){
    outEl.innerHTML = `<div class="pretty"><div class="title"><div class="h">Map</div><div class="sub">not available</div></div>
      <div class="pad muted">No map endpoint found for <span class="mono">${esc(file)}</span>. (If your API serves a map, expose /api/generated/{file}/map.)</div>
    </div>`;
    toast("Map endpoint not found", "warn", 4200);
    return;
  }

  outEl.innerHTML = `
    <div class="pretty">
      <div class="title">
        <div class="h">Map</div>
        <div class="sub">${esc(file)}</div>
      </div>
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

async function doValidate(file, outEl){
  if (!outEl) return;
  outEl.classList.remove("hidden");
  outEl.innerHTML = `<div class="pretty"><div class="title"><div class="h">Validate</div><div class="sub">working…</div></div></div>`;
  try{
    const r = await apiPost(`/generated/${encodeURIComponent(file)}/validate`, {});
    outEl.innerHTML = prettyValidateHTML(r.json ?? {});
  }catch(e){
    outEl.innerHTML = `<div class="pretty"><div class="title"><div class="h">Validate</div><div class="sub">error</div></div><div class="pad mono">${esc(e.message)}</div></div>`;
    toast(`Validate failed: ${e.message}`, "bad", 5200);
  }
}

async function doCompare(file, outEl){
  if (!outEl) return;
  outEl.classList.remove("hidden");
  outEl.innerHTML = `<div class="pretty"><div class="title"><div class="h">Compare</div><div class="sub">building baseline / comparing…</div></div></div>`;
  try{
    const r = await apiPost(`/generated/${encodeURIComponent(file)}/compare`, {});
    outEl.innerHTML = prettyCompareHTML(r.json ?? {});
  }catch(e){
    outEl.innerHTML = `<div class="pretty"><div class="title"><div class="h">Compare</div><div class="sub">error</div></div><div class="pad mono">${esc(e.message)}</div></div>`;
    toast(`Compare failed: ${e.message}`, "bad", 5200);
  }
}

/* ---------- Boot ---------- */

async function refreshTopStatus(){
  const pillHealth = $("#pillHealth");
  const pillRepo = $("#pillRepo");
  const pillTime = $("#pillTime");

  try{
    const { json } = await apiGet("/health");
    if (pillHealth) pillHealth.innerHTML = `API: <b>ok</b>`;
    if (pillRepo) pillRepo.innerHTML = `repo: <b>${esc(json.repo_root || "—")}</b>`;
    if (pillTime) pillTime.innerHTML = `time: <b>${esc((json.time_utc||"").replace("T"," ").replace("+00:00","Z") || "—")}</b>`;
  }catch(e){
    if (pillHealth) pillHealth.innerHTML = `API: <b>down</b>`;
    if (pillRepo) pillRepo.innerHTML = `repo: <b>—</b>`;
    if (pillTime) pillTime.innerHTML = `time: <b>—</b>`;
  }
}

async function refreshCatalogsAndCache(){
  try{
    const c = await apiGet("/catalogs");
    state.catalogs = (c.json?.items || c.json || []);
    renderCatalogs();
  }catch(e){
    state.catalogs = [];
    renderCatalogs();
    toast(`Failed to load catalogs: ${e.message}`, "bad", 5200);
  }

  try{
    const cc = await apiGet("/cache");
    state.cache = cc.json || null;
    renderCache();
  }catch(e){
    state.cache = null;
    renderCache();
    toast(`Failed to load cache: ${e.message}`, "bad", 5200);
  }
}

function bindEvents(){
  const search = $("#accessionSearch");
  if (search){
    search.addEventListener("input", ()=>applyAccFilter());
  }

  const reload = $("#btnReloadAcc");
  if (reload){
    reload.addEventListener("click", ()=>{
      if (state.currentCatalog) selectCatalog(state.currentCatalog);
      else toast("Pick a catalog first", "warn", 2400);
    });
  }

  const more = $("#btnLoadMoreAcc");
  if (more){
    more.addEventListener("click", ()=>loadMoreAccessions());
  }

  const gen = $("#btnGenerate");
  if (gen){
    gen.addEventListener("click", ()=>doGenerate());
  }

  const clear = $("#btnClearGenerated");
  if (clear){
    clear.addEventListener("click", ()=>{
      setGenerated([]);
      toast("Cleared generated list", "good", 2000);
    });
  }
}

async function boot(){
  ensureUI();
  bindEvents();

  state.generated = loadGeneratedFromLS();
  renderGenerated();

  await refreshTopStatus();
  await refreshCatalogsAndCache();

  // restore last catalog/accession (best effort)
  const lastCat = localStorage.getItem(LS.lastCatalog);
  if (lastCat){
    await selectCatalog(lastCat);
  } else {
    // auto-select a small catalog if present
    const small = state.catalogs.find(x => (x.catalog||"").includes("smoke")) || state.catalogs[0];
    if (small?.catalog) await selectCatalog(small.catalog);
  }

  const lastAcc = localStorage.getItem(LS.lastAcc);
  if (lastAcc){
    // only auto-load if it’s in the currently loaded slice
    if (state.accAll.includes(lastAcc)) loadGenomeSummary(lastAcc);
  }
}

document.addEventListener("DOMContentLoaded", boot);
