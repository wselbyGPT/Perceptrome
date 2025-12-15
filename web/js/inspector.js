import { U } from "./utils.js";
import { api } from "./api.js";
import { LS } from "./config.js";
import { Store } from "./store.js";
import { Toasts } from "./toasts.js";

function badge(cls, k, v) {
  return `<div class="bx ${cls}"><span class="k">${U.esc(k)}</span><span class="v">${U.esc(v)}</span></div>`;
}

function prettyGenomeSummaryHTML(s) {
  const acc = s.acc || s.accession || s.id || "—";
  const length = Number(s.length_bp ?? s.length ?? 0);
  const gc = U.pct(s.gc_pct ?? s.gc);
  const np = U.pct(s.n_pct ?? s.n);
  const inv = U.pct(s.invalid_pct ?? s.invalid);

  const feats = s.features || s.feature_counts || {};
  const cds = feats.cds ?? feats.CDS ?? feats.cds_count ?? s.cds_count;
  const gene = feats.gene ?? feats.genes ?? feats.gene_count ?? s.gene_count;
  const trna = feats.trna ?? feats.tRNA ?? feats.trna_count ?? s.trna_count;
  const rrna = feats.rrna ?? feats.rRNA ?? feats.rrna_count ?? s.rrna_count;

  const clsN = np != null && np <= 1 ? "good" : np != null && np <= 5 ? "warn" : "bad";

  return `
    <div class="pretty">
      <div class="title">
        <div class="h">Genome Inspector</div>
        <div class="sub">${U.esc(acc)}</div>
      </div>

      <div class="pretty-grid">
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
        <pre>${U.esc(JSON.stringify(s, null, 2))}</pre>
      </details>
    </div>
  `;
}

function featuresTableHTML(rows) {
  const list = Array.isArray(rows) ? rows : [];
  const max = Math.min(list.length, 500);

  const countsKind = {};
  const countsSrc = {};
  for (const r of list) {
    const k = String(r.kind ?? r.type ?? "—");
    const s = String(r.source ?? "—");
    countsKind[k] = (countsKind[k] || 0) + 1;
    countsSrc[s] = (countsSrc[s] || 0) + 1;
  }

  const chips = [];
  Object.keys(countsKind).sort((a,b)=>countsKind[b]-countsKind[a]).slice(0,8)
    .forEach(k => chips.push(`<span class="gi-chip">${U.esc(k)}: <b>${U.esc(countsKind[k])}</b></span>`));
  Object.keys(countsSrc).sort((a,b)=>countsSrc[b]-countsSrc[a]).slice(0,4)
    .forEach(s => chips.push(`<span class="gi-chip">${U.esc(s)}: <b>${U.esc(countsSrc[s])}</b></span>`));

  const rowsHtml = [];
  for (let i = 0; i < max; i++) {
    const r = list[i] || {};
    const kind = r.kind ?? r.type ?? "—";
    const start = r.start ?? r.begin ?? r.s ?? "—";
    const end = r.end ?? r.stop ?? r.e ?? "—";
    const strand = r.strand === -1 || r.strand === "-1" || r.strand === "-" ? "-" : "+";
    const length = r.length ?? (Number.isFinite(+end) && Number.isFinite(+start) ? +end - +start + 1 : "—");
    const label = r.label ?? r.name ?? r.product ?? r.gene ?? "";
    const source = r.source ?? "—";

    rowsHtml.push(`
      <tr>
        <td><span class="mono" style="font-weight:800">${U.esc(kind)}</span></td>
        <td class="mono">${U.esc(start)}</td>
        <td class="mono">${U.esc(end)}</td>
        <td class="mono">${U.esc(strand)}</td>
        <td class="mono">${U.esc(length)}</td>
        <td>${U.esc(label || "")}</td>
        <td class="mono">${U.esc(source)}</td>
      </tr>
    `);
  }

  const note = list.length > max ? `<div class="muted" style="margin-top:10px;margin-left:0">Showing first <b>${max}</b> of <b>${list.length}</b> features.</div>` : "";
  return `
    <div style="margin-bottom:10px;margin-left:0">${chips.join("") || `<span class="muted">No feature counts.</span>`}</div>
    <div class="gi-table-wrap">
      <table class="gi-table">
        <thead><tr><th>Type</th><th>Start</th><th>End</th><th>Strand</th><th>Len</th><th>Label</th><th>Source</th></tr></thead>
        <tbody>${rowsHtml.join("") || `<tr><td colspan="7" class="muted">No features.</td></tr>`}</tbody>
      </table>
    </div>
    ${note}
  `;
}

async function renderInspectorMap(acc) {
  const host = U.$("#gi-map");
  const pdfBtn = U.$("#gi-pdf");
  if (!host) return;

  const svgPath = `/genome/${encodeURIComponent(acc)}/map.svg`;
  const pdfPath = `/genome/${encodeURIComponent(acc)}/map.pdf`;
  const svgUrl = api.url(svgPath) + `?ts=${Date.now()}`;
  const pdfUrl = api.url(pdfPath);

  try {
    const res = await fetch(svgUrl, { method: "GET" });
    if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
    const txt = await res.text();
    if (!String(txt || "").trim().startsWith("<svg")) throw new Error("Map endpoint did not return SVG.");
    host.innerHTML = txt;

    if (pdfBtn) {
      const ok = await api.endpointExists(pdfUrl);
      pdfBtn.style.display = ok ? "inline-block" : "none";
      if (ok) pdfBtn.href = pdfUrl;
    }
  } catch (e) {
    host.innerHTML = `<div class="muted">No map endpoint found yet for <span class="mono">${U.esc(acc)}</span>.</div>`;
    if (pdfBtn) pdfBtn.style.display = "none";
  }
}

async function renderInspectorFeatures(acc) {
  const host = U.$("#gi-features");
  if (!host) return;

  try {
    const r = await api.get(`/genome/${encodeURIComponent(acc)}/features`);
    const obj = r.json ?? null;
    const rows = Array.isArray(obj) ? obj : obj?.features || obj?.items || obj?.rows || obj?.data || [];
    if (!Array.isArray(rows) || rows.length === 0) {
      host.innerHTML = `<div class="muted">No features returned yet for <span class="mono">${U.esc(acc)}</span>.</div>`;
      return;
    }
    host.innerHTML = featuresTableHTML(rows);
  } catch (e) {
    host.innerHTML = `<div class="muted">No features endpoint found yet for <span class="mono">${U.esc(acc)}</span>.</div>`;
  }
}

export async function loadGenomeInspector(acc) {
  const body = U.$("#inspectorBody");
  const meta = U.$("#inspectorMeta");
  if (meta) meta.textContent = acc;

  if (body) body.innerHTML = `<div class="muted">Loading inspector for <span class="mono">${U.esc(acc)}</span>…</div>`;

  try {
    const { json } = await api.get(`/genome/${encodeURIComponent(acc)}/summary`);
    Store.set(LS.lastAcc, acc);

    const svgOpen = api.url(`/genome/${encodeURIComponent(acc)}/map.svg`);
    const pdfUrl = api.url(`/genome/${encodeURIComponent(acc)}/map.pdf`);

    if (body) {
      body.innerHTML = `
        ${prettyGenomeSummaryHTML(json || {})}
        <div class="pretty" style="margin-top:12px">
          <div class="title"><div class="h">Circular Map</div><div class="sub">SVG preview + PDF export</div></div>
          <div class="pad">
            <div id="gi-map" class="gi-map"><div class="muted">Loading map…</div></div>
            <div class="row" style="padding:10px 0 0;justify-content:flex-end;gap:8px">
              <a class="btn ghost" href="${svgOpen}" target="_blank" rel="noopener">Open SVG</a>
              <a id="gi-pdf" class="btn ghost" href="${pdfUrl}" target="_blank" rel="noopener" style="display:none">Download PDF</a>
            </div>
          </div>
        </div>

        <div class="pretty" style="margin-top:12px">
          <div class="title"><div class="h">Features</div><div class="sub">GenBank annotations or ORF fallback</div></div>
          <div class="pad" id="gi-features"><div class="muted">Loading features…</div></div>
        </div>
      `;
    }

    await Promise.allSettled([renderInspectorMap(acc), renderInspectorFeatures(acc)]);
  } catch (e) {
    if (body) body.innerHTML = `<div class="pretty"><div class="title"><div class="h">Inspector</div><div class="sub">error</div></div><div class="pad mono">${U.esc(e.message)}</div></div>`;
    Toasts.toast(`Inspector failed: ${e.message}`, "bad", 4500);
  }
}
