import { U } from "./utils.js";
import { api } from "./api.js";
import { state } from "./state.js";
import { LS } from "./config.js";
import { Store } from "./store.js";
import { Toasts } from "./toasts.js";
import { loadGenomeInspector } from "./inspector.js";

export function skeletonList(container, count = 8) {
  if (!container) return;
  container.innerHTML = "";
  for (let i = 0; i < count; i++) container.appendChild(U.el("div", { class: "skeleton skeleton-row" }));
}

function parseAccessionsFromResponse(payload) {
  if (!payload) return [];
  if (typeof payload === "string") return payload.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
  if (Array.isArray(payload)) return payload.map(x => (typeof x === "string" ? x : (x.acc || x.accession || x.id || ""))).filter(Boolean);
  const arr = payload.accessions || payload.items || payload.lines || payload.data || null;
  if (Array.isArray(arr)) return arr.map(x => (typeof x === "string" ? x : (x.acc || x.accession || x.id || ""))).filter(Boolean);
  return [];
}

async function fetchCatalogAccessions(catalog, limit = 2000, offset = 0) {
  const tries = [
    `/catalog/${encodeURIComponent(catalog)}?limit=${limit}&offset=${offset}`,
    `/catalogs/${encodeURIComponent(catalog)}?limit=${limit}&offset=${offset}`,
    `/catalog/${encodeURIComponent(catalog)}/accessions?limit=${limit}&offset=${offset}`,
    `/catalog/${encodeURIComponent(catalog)}/preview?limit=${limit}&offset=${offset}`,
    `/catalog/${encodeURIComponent(catalog)}`,
    `/catalogs/${encodeURIComponent(catalog)}`,
  ];
  let lastErr = null;
  for (const path of tries) {
    try {
      const out = await api.get(path);
      const raw = out.json !== null ? out.json : out.text;
      const list = parseAccessionsFromResponse(raw);
      return { list, pathUsed: path };
    } catch (e) { lastErr = e; }
  }
  throw lastErr || new Error("No catalog endpoint worked");
}

export function renderCatalogs() {
  const host = U.$("#catalogList");
  const count = U.$("#catalogCount");
  if (!host) return;

  host.innerHTML = "";
  if (count) count.textContent = state.catalogs.length ? `(${state.catalogs.length})` : "";

  for (const it of state.catalogs) {
    const name = it.catalog || it.name || "—";
    const bytes = U.fmtBytes(it.bytes ?? 0);
    const mtime = U.fmtIso(it.mtime);
    const item = U.el(
      "div",
      { class: "item", role: "button", tabindex: "0" },
      U.el("div", { class: "l" }, U.el("div", { class: "mono", text: name }), U.el("div", { class: "muted", text: it.name || "" })),
      U.el("div", { class: "r" }, U.el("div", { class: "muted", text: bytes }), U.el("div", { class: "muted", text: mtime }))
    );
    item.addEventListener("click", () => selectCatalog(name));
    item.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectCatalog(name); } });
    host.appendChild(item);
  }
}

export function renderCache() {
  const host = U.$("#cacheList");
  const sum = U.$("#cacheSummary");
  if (!host) return;

  host.innerHTML = "";
  const c = state.cache;
  if (!c) { if (sum) sum.textContent = ""; return; }

  const parts = [];
  for (const key of ["genbank", "fasta", "encoded"]) {
    const b = c[key];
    if (b) parts.push(`${key}:${b.count ?? 0}`);
  }
  if (sum) sum.textContent = parts.length ? `(${parts.join(" • ")})` : "";

  for (const key of ["genbank", "fasta", "encoded"]) {
    const b = c[key];
    if (!b) continue;
    host.appendChild(
      U.el("div", { class: "item" },
        U.el("div", { class: "l" }, U.el("div", { class: "mono", text: key }), U.el("div", { class: "muted", text: b.path || "" })),
        U.el("div", { class: "r" }, U.el("div", { class: "muted", text: (b.count ?? 0).toLocaleString() }), U.el("div", { class: "muted", text: b.exists ? "ok" : "missing" }))
      )
    );
  }
}

export function renderAccessions() {
  const host = U.$("#accessionList");
  const meta = U.$("#accMeta");
  if (!host) return;

  const total = state.accAll.length;
  const shown = state.accFiltered.length;
  if (meta) meta.textContent = state.currentCatalog ? `${state.currentCatalog} • ${shown.toLocaleString()} shown / ${total.toLocaleString()} loaded` : "";

  host.innerHTML = "";
  const maxShow = Math.min(shown, 2500);

  for (let i = 0; i < maxShow; i++) {
    const acc = state.accFiltered[i];
    const item = U.el("div", { class: "item", role: "button", tabindex: "0" },
      U.el("div", { class: "l" }, U.el("div", { class: "mono", text: acc }), U.el("div", { class: "muted", text: "click → inspector" })),
      U.el("div", { class: "r" }, U.el("div", { class: "muted", text: "genome" }), U.el("div", { class: "muted", text: "summary/map/features" }))
    );
    item.addEventListener("click", () => loadGenomeInspector(acc));
    item.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); loadGenomeInspector(acc); } });
    host.appendChild(item);
  }

  if (shown > maxShow) host.appendChild(U.el("div", { class: "pad muted", html: `Showing first <b>${maxShow.toLocaleString()}</b> of <b>${shown.toLocaleString()}</b> (filter to narrow).` }));
}

export function applyAccFilter() {
  const q = (U.$("#accessionSearch")?.value || "").trim().toLowerCase();
  state.accFiltered = !q ? state.accAll.slice() : state.accAll.filter(a => a.toLowerCase().includes(q));
  renderAccessions();
}

export async function selectCatalog(catalog) {
  state.currentCatalog = catalog;
  state.accAll = [];
  state.accFiltered = [];
  state.accOffset = 0;

  Store.set(LS.lastCatalog, catalog);
  Toasts.toast(`Loading catalog: ${catalog}`, "info", 1800);

  const host = U.$("#accessionList");
  if (host) skeletonList(host, 10);

  try {
    const { list, pathUsed } = await fetchCatalogAccessions(catalog, state.accLimit, state.accOffset);
    state.accAll = list;
    state.accOffset += list.length;
    applyAccFilter();
    Toasts.toast(`Loaded ${list.length.toLocaleString()} accessions (${pathUsed})`, "good", 2600);
  } catch (e) {
    state.accAll = [];
    state.accFiltered = [];
    renderAccessions();
    Toasts.toast(`Catalog load failed: ${e.message}`, "bad", 5200);
  }
}

export async function loadMoreAccessions() {
  if (!state.currentCatalog) return Toasts.toast("Pick a catalog first", "warn", 2600);
  try {
    const { list } = await fetchCatalogAccessions(state.currentCatalog, state.accLimit, state.accOffset);
    if (!list.length) return Toasts.toast("No more returned (or API doesn’t paginate)", "warn", 3200);
    state.accAll.push(...list);
    state.accOffset += list.length;
    applyAccFilter();
    Toasts.toast(`+${list.length.toLocaleString()} loaded`, "good", 2400);
  } catch (e) {
    Toasts.toast(`Load more failed: ${e.message}`, "bad", 5200);
  }
}
