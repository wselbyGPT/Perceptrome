import { U } from "./utils.js";

export function ensureUI() {
  const root = U.$("#app") || document.body;

  const hasCore = U.$("#catalogList") && U.$("#accessionList") && U.$("#inspector");
  if (hasCore) return;

  root.innerHTML = "";
  root.appendChild(
    U.el(
      "div",
      { class: "wrap" },
      U.el(
        "header",
        { class: "top" },
        U.el(
          "div",
          { class: "brand" },
          U.el("div", { class: "dot" }),
          U.el("div", { class: "title", html: `<b>Perceptrome</b> <span class="muted">Genome Dashboard</span>` })
        ),
        U.el(
          "div",
          { class: "statusline" },
          U.el("span", { class: "pill", id: "pillHealth", html: "API: <b>…</b>" }),
          U.el("span", { class: "pill", id: "pillRepo", html: "repo: <b>…</b>" }),
          U.el("span", { class: "pill", id: "pillTime", html: "time: <b>…</b>" })
        )
      ),

      U.el(
        "nav",
        { class: "tabs", id: "mainTabs", role: "tablist", "aria-label": "Perceptrome sections" },
        U.el("button", { class: "tab active", id: "tabView", type: "button", role: "tab", "aria-selected": "true", "data-tab": "view", text: "Generate & View" }),
        U.el("button", { class: "tab", id: "tabTrain", type: "button", role: "tab", "aria-selected": "false", "data-tab": "train", text: "Training" })
      ),

      U.el(
        "section",
        { class: "tabpane active", id: "paneView", role: "tabpanel", "aria-labelledby": "tabView" },
        U.el(
          "div",
          { class: "layout-grid" },
          U.el(
            "aside",
            { class: "side" },
            U.el("section", { class: "card" }, U.el("div", { class: "card-h", html: `<b>Catalogs</b><span class="muted" id="catalogCount"></span>` }), U.el("div", { class: "list", id: "catalogList" })),
            U.el("section", { class: "card" }, U.el("div", { class: "card-h", html: `<b>Cache</b><span class="muted" id: "cacheSummary"></span>` }), U.el("div", { class: "list", id: "cacheList" }))
          ),

          U.el(
            "main",
            { class: "main" },
            U.el(
              "section",
              { class: "card" },
              U.el("div", { class: "card-h", html: `<b>Accessions</b><span class="muted" id="accMeta"></span>` }),
              U.el(
                "div",
                { class: "row" },
                U.el("input", { id: "accessionSearch", class: "input", placeholder: "search accessions (filter)…" }),
                U.el("button", { id: "btnReloadAcc", class: "btn", text: "Reload" }),
                U.el("button", { id: "btnLoadMoreAcc", class: "btn ghost", text: "Load more" })
              ),
              U.el("div", { class: "list tall", id: "accessionList" })
            ),

            U.el(
              "section",
              { class: "card" },
              U.el("div", { class: "card-h", html: `<b>Generate</b><span class="muted">→ Map / Validate / Compare</span>` }),
              U.el(
                "div",
                { class: "row wraprow" },
                U.el("select", { id: "genMode", class: "input" }, U.el("option", { value: "genome", text: "genome" }), U.el("option", { value: "protein", text: "protein" })),
                U.el("input", { id: "genLen", class: "input", placeholder: "length (bp or aa)", value: "20000" }),
                U.el("input", { id: "genGc", class: "input", placeholder: "GC target (0-1 or %)", value: "0.50" }),
                U.el("input", { id: "genSeed", class: "input", placeholder: "seed (optional)" }),
                U.el("button", { id: "btnGenerate", class: "btn", text: "Generate" })
              ),
              U.el("div", { class: "list", id: "generatedList" }),
              U.el("div", { class: "row wraprow" }, U.el("button", { id: "btnClearGenerated", class: "btn ghost", text: "Clear list" }))
            )
          ),

          U.el(
            "aside",
            { class: "inspect" },
            U.el(
              "section",
              { class: "card", id: "inspector" },
              U.el("div", { class: "card-h", html: `<b>Genome Inspector</b><span class="muted" id="inspectorMeta"></span>` }),
              U.el("div", { id: "inspectorBody", class: "pad", html: `<div class="muted">Click an accession to view summary + circular map + features.</div>` })
            ),
            U.el(
              "section",
              { class: "card" },
              U.el("div", { class: "card-h", html: `<b>Outputs</b><span class="muted">map / validate / compare</span>` }),
              U.el("div", { id: "mapOut", class: "pad hidden" }),
              U.el("div", { id: "validateOut", class: "pad hidden" }),
              U.el("div", { id: "compareOut", class: "pad hidden" })
            )
          )
        )
      ),

      U.el(
        "section",
        { class: "tabpane", id: "paneTrain", role: "tabpanel", "aria-labelledby": "tabTrain" },
        U.el(
          "div",
          { class: "layout-grid2" },
          U.el(
            "div",
            {},
            U.el(
              "section",
              { class: "card" },
              U.el("div", { class: "card-h", html: `<b>Training Runner</b><span class="muted">start a job</span>` }),
              U.el("div", { class: "row wraprow" },
                U.el("select", { id: "trainCatalog", class: "input" }, U.el("option", { value: "", text: "(select dataset catalog)" })),
                U.el("select", { id: "trainKind", class: "input" }, U.el("option", { value: "genome", text: "genome" }), U.el("option", { value: "protein", text: "protein" })),
                U.el("input", { id: "trainRunName", class: "input", placeholder: "run name (optional)" })
              ),
              U.el("div", { class: "row wraprow" },
                U.el("input", { id: "trainSteps", class: "input", placeholder: "steps/epochs (optional)" }),
                U.el("input", { id: "trainBatch", class: "input", placeholder: "batch_size (optional)" }),
                U.el("input", { id: "trainLr", class: "input", placeholder: "learning_rate (optional)" }),
                U.el("select", { id: "trainDevice", class: "input" },
                  U.el("option", { value: "auto", text: "device: auto" }),
                  U.el("option", { value: "cpu", text: "device: cpu" }),
                  U.el("option", { value: "cuda", text: "device: cuda" })
                )
              ),
              U.el("div", { class: "row wraprow" },
                U.el("input", { id: "trainRunId", class: "input", placeholder: "run_id (optional, for stop)" }),
                U.el("button", { id: "btnTrainStart", class: "btn", text: "Start" }),
                U.el("button", { id: "btnTrainStop", class: "btn ghost", text: "Stop" }),
                U.el("button", { id: "btnTrainRefresh", class: "btn ghost", text: "Refresh" })
              )
            ),
            U.el("section", { class: "card" }, U.el("div", { class: "card-h", html: `<b>Recent Runs</b><span class="muted">from status</span>` }), U.el("div", { class: "list", id: "trainJobs" }, U.el("div", { class: "pad muted", text: "No training runs loaded yet." })))
          ),
          U.el(
            "div",
            {},
            U.el("section", { class: "card" }, U.el("div", { class: "card-h", html: `<b>Status</b><span class="muted">best-effort</span>` }), U.el("div", { id: "trainStatus", class: "pad", html: `<div class="muted">Click <b>Refresh</b> to load training status.</div>` })),
            U.el("section", { class: "card" }, U.el("div", { class: "card-h", html: `<b>Logs</b><span class="muted">tail</span>` }), U.el("div", { class: "pad" }, U.el("pre", { id: "trainLogs", class: "logbox", text: "(no logs loaded)" })))
          )
        )
      )
    )
  );
}
