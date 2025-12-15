export const API_PREFIX = "/api";

export const LS = {
  generated: "perceptrome.generated.v1",
  lastCatalog: "perceptrome.lastCatalog.v1",
  lastAcc: "perceptrome.lastAcc.v1",
  lastTab: "perceptrome.lastTab.v1",
};

export const TRAIN_ENDPOINTS = {
  status: ["/training/status", "/train/status", "/training", "/train"],
  start: ["/training/start", "/train/start", "/training/run", "/train/run", "/training", "/train"],
  stop: ["/training/stop", "/train/stop", "/training/cancel", "/train/cancel"],
  logs: ["/training/logs", "/train/logs", "/training/log", "/train/log"],
};
