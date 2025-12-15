import { U } from "./utils.js";
import { api } from "./api.js";
import { state } from "./state.js";
import { Toasts } from "./toasts.js";
import { renderCatalogs, renderCache } from "./catalogs.js";
import { syncTrainingCatalogOptions } from "./training.js";

export async function refreshTopStatus() {
  const pillHealth = U.$("#pillHealth");
  const pillRepo = U.$("#pillRepo");
  const pillTime = U.$("#pillTime");

  try {
    const { json } = await api.get("/health");
    if (pillHealth) pillHealth.innerHTML = `API: <b>ok</b>`;
    if (pillRepo) pillRepo.innerHTML = `repo: <b>${U.esc(json.repo_root || "—")}</b>`;
    if (pillTime) pillTime.innerHTML = `time: <b>${U.esc((json.time_utc || "").replace("T", " ").replace("+00:00", "Z") || "—")}</b>`;
  } catch {
    if (pillHealth) pillHealth.innerHTML = `API: <b>down</b>`;
    if (pillRepo) pillRepo.innerHTML = `repo: <b>—</b>`;
    if (pillTime) pillTime.innerHTML = `time: <b>—</b>`;
  }
}

export async function refreshCatalogsAndCache() {
  try {
    const c = await api.get("/catalogs");
    state.catalogs = c.json?.items || c.json || [];
    renderCatalogs();
    syncTrainingCatalogOptions();
  } catch (e) {
    state.catalogs = [];
    renderCatalogs();
    Toasts.toast(`Failed to load catalogs: ${e.message}`, "bad", 5200);
  }

  try {
    const cc = await api.get("/cache");
    state.cache = cc.json || null;
    renderCache();
  } catch (e) {
    state.cache = null;
    renderCache();
    Toasts.toast(`Failed to load cache: ${e.message}`, "bad", 5200);
  }
}
