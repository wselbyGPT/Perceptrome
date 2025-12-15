import { U } from "./utils.js";
import { LS } from "./config.js";
import { Store } from "./store.js";
import { state } from "./state.js";

import { ensureUI } from "./ui.js";
import { initTabs, setTab } from "./tabs.js";
import { refreshTopStatus, refreshCatalogsAndCache } from "./topstatus.js";
import { applyAccFilter, selectCatalog, loadMoreAccessions } from "./catalogs.js";
import { loadGenomeInspector } from "./inspector.js";
import { loadGeneratedFromLS, renderGenerated, setGenerated, doGenerate } from "./generate.js";
import { refreshTraining, startTraining, stopTraining } from "./training.js";
import { Toasts } from "./toasts.js";

export function bindEvents() {
  // Tabs
  U.$("#tabView")?.addEventListener("click", () => setTab("view"));
  U.$("#tabTrain")?.addEventListener("click", () => setTab("train"));
  window.addEventListener("hashchange", () => {
    const h = (location.hash || "").replace("#", "").trim().toLowerCase();
    if (h === "train" || h === "view") setTab(h, { setHash: false });
  });

  // Training
  U.$("#btnTrainStart")?.addEventListener("click", () => startTraining());
  U.$("#btnTrainStop")?.addEventListener("click", () => stopTraining());
  U.$("#btnTrainRefresh")?.addEventListener("click", () => refreshTraining());

  // Accessions filter
  const search = U.$("#accessionSearch");
  if (search) search.addEventListener("input", U.debounce(() => applyAccFilter(), 120));

  U.$("#btnReloadAcc")?.addEventListener("click", () => {
    const cat = state.currentCatalog;
    if (cat) selectCatalog(cat);
    else Toasts.toast("Pick a catalog first", "warn", 2400);
  });
  U.$("#btnLoadMoreAcc")?.addEventListener("click", () => loadMoreAccessions());

  // Generate
  U.$("#btnGenerate")?.addEventListener("click", () => doGenerate());
  U.$("#btnClearGenerated")?.addEventListener("click", () => setGenerated([]));
}

export async function boot() {
  ensureUI();
  bindEvents();
  initTabs();

  state.generated = loadGeneratedFromLS();
  renderGenerated();

  await refreshTopStatus();
  await refreshCatalogsAndCache();

  // restore last catalog/accession
  const lastCat = Store.get(LS.lastCatalog);
  if (lastCat) await selectCatalog(lastCat);

  const lastAcc = Store.get(LS.lastAcc);
  if (lastAcc && state.accAll.includes(lastAcc)) loadGenomeInspector(lastAcc);
}
