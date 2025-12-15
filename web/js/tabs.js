import { U } from "./utils.js";
import { LS } from "./config.js";
import { Store } from "./store.js";
import { refreshTraining } from "./training.js";

export function setTab(tab, opts = { setHash: true }) {
  tab = tab === "train" ? "train" : "view";
  const isTrain = tab === "train";

  const bView = U.$("#tabView");
  const bTrain = U.$("#tabTrain");
  const pView = U.$("#paneView");
  const pTrain = U.$("#paneTrain");

  if (bView) { bView.classList.toggle("active", !isTrain); bView.setAttribute("aria-selected", (!isTrain).toString()); }
  if (bTrain) { bTrain.classList.toggle("active", isTrain); bTrain.setAttribute("aria-selected", isTrain.toString()); }
  if (pView) pView.classList.toggle("active", !isTrain);
  if (pTrain) pTrain.classList.toggle("active", isTrain);

  Store.set(LS.lastTab, tab);
  if (opts?.setHash) {
    const h = "#" + tab;
    if (location.hash !== h) history.replaceState(null, "", h);
  }
  if (isTrain) refreshTraining().catch(() => {});
}

export function initTabs() {
  const fromHash = (location.hash || "").replace("#", "").trim().toLowerCase();
  const saved = Store.get(LS.lastTab);
  const tab = fromHash === "train" || fromHash === "view" ? fromHash : saved || "view";
  setTab(tab, { setHash: false });
}
