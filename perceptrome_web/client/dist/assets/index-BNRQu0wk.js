(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const s of document.querySelectorAll('link[rel="modulepreload"]'))l(s);new MutationObserver(s=>{for(const c of s)if(c.type==="childList")for(const u of c.addedNodes)u.tagName==="LINK"&&u.rel==="modulepreload"&&l(u)}).observe(document,{childList:!0,subtree:!0});function a(s){const c={};return s.integrity&&(c.integrity=s.integrity),s.referrerPolicy&&(c.referrerPolicy=s.referrerPolicy),s.crossOrigin==="use-credentials"?c.credentials="include":s.crossOrigin==="anonymous"?c.credentials="omit":c.credentials="same-origin",c}function l(s){if(s.ep)return;s.ep=!0;const c=a(s);fetch(s.href,c)}})();const C=document.querySelector("#app");C.innerHTML=`
  <div class="app-root">
    <header class="app-header">
      <div class="app-title">perceptrome</div>
    </header>

    <div class="app-body">
      <nav class="tabs" aria-label="Main">
        <button class="tab tab--active" data-tab="config">Home / Config</button>
        <button class="tab" data-tab="train">Train</button>
        <button class="tab" data-tab="generate">Generate</button>
        <button class="tab" data-tab="view">View</button>
        <button class="tab" data-tab="history">History</button>
      </nav>

      <main class="tab-panels">
        <!-- Home / Config -->
        <section class="tab-panel tab-panel--active" data-tab-panel="config">
          <div class="panel-group">
            <section class="panel">
              <h2 class="panel-title">Project settings</h2>
              <div class="form-grid">
                <label class="field">
                  <span class="field-label">Project dir:</span>
                  <input class="field-input" type="text" value="." />
                </label>

                <label class="field">
                  <span class="field-label">stream_config.yaml:</span>
                  <input class="field-input" type="text" value="stream_config.yaml" />
                </label>

                <label class="field">
                  <span class="field-label">Dataset list file:</span>
                  <input class="field-input" type="text" data-action="dataset-list-file" value="config/plasmids_10.txt" />
                </label>

                <label class="field">
                  <span class="field-label">Epochs:</span>
                  <input class="field-input" type="number" value="10" />
                </label>

                <label class="field">
                  <span class="field-label">Batch size:</span>
                  <input class="field-input" type="number" value="256" />
                </label>

                <label class="field">
                  <span class="field-label">Learning rate:</span>
                  <input class="field-input" type="number" step="0.000001" value="0.001" />
                </label>
              </div>
            </section>

            <section class="panel">
              <h2 class="panel-title">Custom dataset builder</h2>

              <div class="form-grid form-grid--two">
                <label class="field">
                  <span class="field-label">Category:</span>
                  <select class="field-input" data-action="quota-category">
                    <option>Plasmids</option>
                    <option>Bacterial genomes</option>
                    <option>Custom</option>
                  </select>
                </label>

                <label class="field">
                  <span class="field-label">Source catalog:</span>
                  <input class="field-input" type="text" data-action="quota-source" value="config/plasmids_10.txt" />
                </label>

                <label class="field">
                  <span class="field-label">Count:</span>
                  <input class="field-input" type="number" data-action="quota-count" value="100" />
                </label>
              </div>

              <label class="checkbox">
                <input type="checkbox" data-action="shuffle-categories" checked />
                <span>Shuffle inside each category</span>
              </label>

              <div class="button-row">
                <button class="btn btn-secondary" type="button" data-action="add-quota-row">Add category quota</button>
                <button class="btn btn-secondary" type="button" data-action="remove-quota-row">Remove selected</button>
              </div>

              <div class="table-wrapper">
                <table class="table">
                  <thead>
                    <tr>
                      <th>Category</th>
                      <th>Source</th>
                      <th>Count</th>
                    </tr>
                  </thead>
                  <tbody data-action="quota-table-body">
                    <!-- Empty state, matches Qt view -->
                  </tbody>
                </table>
              </div>

              <div class="form-grid">
                <label class="field">
                  <span class="field-label">Output catalog:</span>
                  <input class="field-input" type="text" data-action="output-catalog" value="config/custom_dataset.txt" />
                </label>
              </div>

              <div class="button-row">
                <button class="btn" type="button" data-action="create-dataset-list">Create dataset list</button>
              </div>

              <div class="button-row button-row--end">
                <button class="btn btn-secondary" type="button">Save config</button>
                <button class="btn" type="button" data-action="go-train">Go to Train tab</button>
              </div>

              <p class="panel-footer-text" data-action="config-status">Loaded saved config (if any).</p>
            </section>
          </div>
        </section>

        <!-- Train -->
        <section class="tab-panel" data-tab-panel="train">
          <section class="panel">
            <h2 class="panel-title">Training setup</h2>

            <div class="form-grid">
              <label class="field">
                <span class="field-label">Neural network:</span>
                <select class="field-input">
                  <option>mlp</option>
                  <option>cnn</option>
                  <option>transformer</option>
                </select>
              </label>

              <label class="field field--full">
                <span class="field-label">Training command:</span>
                <input
                  class="field-input"
                  type="text"
                  data-action="train-command"
                  value="perceptrome --config config/stream_config.yaml stream --catalog config/plasmids_10.txt --model-type mlp --steps-per-plasmid 10 --batch-size 256"
                />
              </label>
            </div>

            <div class="button-row">
              <button class="btn btn-secondary" type="button">Help</button>
              <button class="btn btn-secondary" type="button">Build command</button>
              <button class="btn" type="button" data-action="train-start">Start</button>
              <button class="btn btn-danger" type="button" data-action="train-stop" disabled>Stop</button>
            </div>

            <div class="progress-bar">
              <div class="progress-bar__track">
                <div class="progress-bar__value" style="width: 0%;"></div>
              </div>
              <div class="progress-bar__label">0%</div>
            </div>

            <div class="log-area">
              <div class="log-area__label">Live training output will appear here...</div>
              <pre class="log-area__body" data-action="train-log"></pre>
            </div>
          </section>
        </section>

        <!-- Generate -->
        <section class="tab-panel" data-tab-panel="generate">
          <section class="panel">
            <h2 class="panel-title">Generation setup</h2>

            <div class="form-grid">
              <label class="field">
                <span class="field-label">Trained model:</span>
                <select class="field-input">
                  <option>model/checkpoints/latest.pt</option>
                </select>
              </label>

              <label class="field field--full">
                <span class="field-label">Generate command:</span>
                <input
                  class="field-input"
                  type="text"
                  data-action="generate-command"
                  value="perceptrome --config config/stream_config.yaml generate-plasmid --length-bp 10000 --output generated/novel_plasmid.fasta"
                />
              </label>
            </div>

            <div class="button-row">
              <button class="btn btn-secondary" type="button">Help</button>
              <button class="btn btn-secondary" type="button">Refresh models</button>
              <button class="btn btn-secondary" type="button">Build command</button>
              <button class="btn" type="button" data-action="generate-start">Start</button>
              <button class="btn btn-danger" type="button" data-action="generate-stop" disabled>Stop</button>
            </div>

            <div class="progress-bar">
              <div class="progress-bar__track">
                <div class="progress-bar__value" style="width: 0%;"></div>
              </div>
              <div class="progress-bar__label">0%</div>
            </div>

            <div class="log-area">
              <div class="log-area__label">Live generate output will appear here...</div>
              <pre class="log-area__body" data-action="generate-log"></pre>
            </div>
          </section>
        </section>

        <!-- View -->
        <section class="tab-panel" data-tab-panel="view">
          <div class="panel-group">
            <section class="panel">
              <h2 class="panel-title">Genome source</h2>

              <div class="form-grid">
                <label class="field">
                  <span class="field-label">Genome accession:</span>
                  <input class="field-input" type="text" placeholder="Example: NC_000913.3" />
                </label>

                <label class="field">
                  <span class="field-label">FASTA path:</span>
                  <input class="field-input" type="text" value="generated/novel_plasmid.fasta" />
                </label>
              </div>
            </section>

            <section class="panel">
              <h2 class="panel-title">PDF output</h2>

              <div class="form-grid form-grid--two">
                <label class="field">
                  <span class="field-label">Render mode:</span>
                  <select class="field-input">
                    <option>Circular</option>
                    <option>Linear</option>
                  </select>
                </label>

                <label class="field">
                  <span class="field-label">Output PDF:</span>
                  <input class="field-input" type="text" value="generated/circular_genome.pdf" />
                </label>

                <label class="field field--full">
                  <span class="field-label">Title:</span>
                  <input class="field-input" type="text" placeholder="Optional title override" />
                </label>
              </div>

              <div class="button-row">
                <button class="btn" type="button">Generate PDF</button>
                <button class="btn btn-secondary" type="button">Open PDF Window</button>
              </div>

              <div class="log-area">
                <div class="log-area__label">PDF generation status will appear here...</div>
                <pre class="log-area__body"></pre>
              </div>
            </section>
          </div>
        </section>

        <!-- History -->
        <section class="tab-panel" data-tab-panel="history">
          <section class="panel">
            <div class="panel-header-row">
              <h2 class="panel-title">History</h2>
              <button class="btn btn-secondary" type="button">Clear history</button>
            </div>

            <div class="table-wrapper">
              <table class="table">
                <thead>
                  <tr>
                    <th style="width: 20%;">Time</th>
                    <th style="width: 20%;">Action</th>
                    <th>Details</th>
                  </tr>
                </thead>
                <tbody>
                  <!-- Empty, matches Qt screenshot -->
                </tbody>
              </table>
            </div>
          </section>
        </section>
      </main>
    </div>
  </div>
`;const S=Array.from(document.querySelectorAll(".tab")),k=Array.from(document.querySelectorAll(".tab-panel"));function _(t){S.forEach(e=>{const a=e.dataset.tab===t;e.classList.toggle("tab--active",a)}),k.forEach(e=>{const a=e.dataset.tabPanel===t;e.classList.toggle("tab-panel--active",a)})}S.forEach(t=>{t.addEventListener("click",()=>{const e=t.dataset.tab;_(e)})});const m=document.querySelector('[data-action="go-train"]');m&&m.addEventListener("click",()=>_("train"));const x=document.querySelector('[data-action="train-command"]'),N=document.querySelector('[data-action="generate-command"]'),p=document.querySelector('[data-action="train-start"]'),b=document.querySelector('[data-action="train-stop"]'),f=document.querySelector('[data-action="generate-start"]'),g=document.querySelector('[data-action="generate-stop"]'),O=document.querySelector('[data-action="train-log"]'),A=document.querySelector('[data-action="generate-log"]'),h=document.querySelector('[data-action="dataset-list-file"]'),P=document.querySelector('[data-action="quota-category"]'),B=document.querySelector('[data-action="quota-source"]'),T=document.querySelector('[data-action="quota-count"]'),R=document.querySelector('[data-action="shuffle-categories"]'),$=document.querySelector('[data-action="add-quota-row"]'),F=document.querySelector('[data-action="remove-quota-row"]'),d=document.querySelector('[data-action="quota-table-body"]'),y=document.querySelector('[data-action="output-catalog"]'),H=document.querySelector('[data-action="create-dataset-list"]'),w=document.querySelector('[data-action="config-status"]');let i=null,n=null;function r(t){w&&(w.textContent=t)}function G(){return d?.querySelector('tr[data-action="quota-row"].is-selected')??null}function Q(){d?.querySelectorAll('tr[data-action="quota-row"]').forEach(t=>{t.classList.remove("is-selected")})}function D(t){if(!d)return;const e=document.createElement("tr");e.dataset.action="quota-row",e.dataset.category=t.category,e.dataset.source=t.source,e.dataset.count=String(t.count),e.innerHTML=`<td>${t.category}</td><td>${t.source}</td><td>${t.count}</td>`,e.addEventListener("click",()=>{const a=e.classList.contains("is-selected");Q(),a||e.classList.add("is-selected")}),d.appendChild(e)}function W(){const t=d?.querySelectorAll('tr[data-action="quota-row"]')??[];return Array.from(t).map(e=>({category:e.dataset.category??"",source:e.dataset.source??"",count:Number(e.dataset.count??"0")}))}function o(t,e){const a=t==="train"?O:A;a&&(a.textContent+=`${e}
`,a.scrollTop=a.scrollHeight)}function v(t,e){t==="train"?(p&&(p.disabled=e),b&&(b.disabled=!e)):(f&&(f.disabled=e),g&&(g.disabled=!e))}function q(){if(i&&(i.readyState===WebSocket.OPEN||i.readyState===WebSocket.CONNECTING))return i;const t=window.location.protocol==="https:"?"wss":"ws";return i=new WebSocket(`${t}://${window.location.host}/ws`),i.addEventListener("open",()=>{o("train","[web] Connected to backend."),o("generate","[web] Connected to backend.")}),i.addEventListener("close",()=>{o("train","[web] Backend connection closed."),o("generate","[web] Backend connection closed."),n&&(v(n,!1),n=null)}),i.addEventListener("message",e=>{let a;try{a=JSON.parse(String(e.data))}catch{if(!n)return;o(n,`[web] Invalid backend message: ${String(e.data)}`);return}if(a.type==="create_dataset_result"){if(a.payload.logs?.length&&r(a.payload.logs.join(" | ")),a.payload.ok&&a.payload.output_catalog){y&&(y.value=a.payload.output_catalog),h&&(h.value=a.payload.output_catalog);const l=a.payload.selected_count??0;r(`Created dataset list with ${l} accessions at ${a.payload.output_catalog}.`)}else r(`Failed to create dataset list: ${a.payload.error??"unknown error"}`);return}if(n){if(a.type==="log"){o(n,a.message);return}if(a.type==="status"){(a.status==="done"||a.status==="error"||a.status==="pending")&&(v(n,!1),n=null);return}if(a.type==="result"){const l=a.payload.exit_code;o(n,`[web] Run finished (exit code: ${l??"unknown"}).`)}}}),i}function E(t,e){const a=q();if(!e.trim()){o(t,"[web] Command is empty.");return}if(n){o(t,"[web] Another run is active. Stop it before starting a new one.");return}n=t,v(t,!0),o(t,`[web] Starting: ${e}`);const l=()=>{a.send(JSON.stringify({type:"start_run",command:e,cwd:"."}))};a.readyState===WebSocket.OPEN?l():a.addEventListener("open",l,{once:!0})}function L(t){if(!i||i.readyState!==WebSocket.OPEN){o(t,"[web] Backend is not connected.");return}if(n!==t){o(t,"[web] No active run in this tab.");return}i.send(JSON.stringify({type:"stop_run"})),o(t,"[web] Stop requested.")}p?.addEventListener("click",()=>E("train",x?.value??""));b?.addEventListener("click",()=>L("train"));f?.addEventListener("click",()=>E("generate",N?.value??""));g?.addEventListener("click",()=>L("generate"));$?.addEventListener("click",()=>{const t=P?.value?.trim()??"",e=B?.value?.trim()??"",a=Number(T?.value??"0");if(!t||!e||!Number.isFinite(a)||a<=0){r("Please provide category, source catalog, and count > 0 before adding a quota row.");return}D({category:t,source:e,count:Math.trunc(a)}),r(`Added quota row for ${t}.`)});F?.addEventListener("click",()=>{const t=G();if(!t){r("Select a quota row to remove.");return}t.remove(),r("Removed selected quota row.")});H?.addEventListener("click",()=>{const t=q(),e=W();if(!e.length){r("Add at least one category quota row before creating a dataset list.");return}const a=y?.value?.trim()??"";if(!a){r("Output catalog path is required.");return}const l=()=>{t.send(JSON.stringify({type:"create_dataset",payload:{category_quotas:e,selected_category_sources:e.map(s=>({category:s.category,source:s.source})),shuffle_within_category:!!R?.checked,output_catalog:a}})),r("Creating dataset list...")};t.readyState===WebSocket.OPEN?l():t.addEventListener("open",l,{once:!0})});
