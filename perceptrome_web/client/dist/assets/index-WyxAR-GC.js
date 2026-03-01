(function(){const a=document.createElement("link").relList;if(a&&a.supports&&a.supports("modulepreload"))return;for(const n of document.querySelectorAll('link[rel="modulepreload"]'))l(n);new MutationObserver(n=>{for(const c of n)if(c.type==="childList")for(const v of c.addedNodes)v.tagName==="LINK"&&v.rel==="modulepreload"&&l(v)}).observe(document,{childList:!0,subtree:!0});function t(n){const c={};return n.integrity&&(c.integrity=n.integrity),n.referrerPolicy&&(c.referrerPolicy=n.referrerPolicy),n.crossOrigin==="use-credentials"?c.credentials="include":n.crossOrigin==="anonymous"?c.credentials="omit":c.credentials="same-origin",c}function l(n){if(n.ep)return;n.ep=!0;const c=t(n);fetch(n.href,c)}})();const R=document.querySelector("#app");R.innerHTML=`
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
                  <input class="field-input" type="text" data-action="view-accession" placeholder="Example: NC_000913.3" />
                </label>

                <label class="field">
                  <span class="field-label">FASTA path:</span>
                  <input class="field-input" type="text" data-action="view-fasta-path" value="generated/novel_plasmid.fasta" />
                </label>
              </div>
            </section>

            <section class="panel">
              <h2 class="panel-title">PDF output</h2>

              <div class="form-grid form-grid--two">
                <label class="field">
                  <span class="field-label">Render mode:</span>
                  <select class="field-input" data-action="view-render-mode">
                    <option value="circular">Circular</option>
                    <option value="linear">Linear</option>
                  </select>
                </label>

                <label class="field">
                  <span class="field-label">Output PDF:</span>
                  <input class="field-input" type="text" data-action="view-output-path" value="generated/circular_genome.pdf" />
                </label>

                <label class="field field--full">
                  <span class="field-label">Title:</span>
                  <input class="field-input" type="text" data-action="view-title" placeholder="Optional title override" />
                </label>
              </div>

              <div class="button-row">
                <button class="btn" type="button" data-action="view-generate-pdf">Generate PDF</button>
                <button class="btn btn-secondary" type="button" data-action="view-open-pdf" disabled>Open / Download PDF</button>
              </div>

              <div class="log-area">
                <div class="log-area__label">PDF generation status will appear here...</div>
                <pre class="log-area__body" data-action="view-log"></pre>
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
`;const x=Array.from(document.querySelectorAll(".tab")),T=Array.from(document.querySelectorAll(".tab-panel"));function N(e){x.forEach(a=>{const t=a.dataset.tab===e;a.classList.toggle("tab--active",t)}),T.forEach(a=>{const t=a.dataset.tabPanel===e;a.classList.toggle("tab-panel--active",t)})}x.forEach(e=>{e.addEventListener("click",()=>{const a=e.dataset.tab;N(a)})});const C=document.querySelector('[data-action="go-train"]');C&&C.addEventListener("click",()=>N("train"));const F=document.querySelector('[data-action="train-command"]'),$=document.querySelector('[data-action="generate-command"]'),y=document.querySelector('[data-action="train-start"]'),m=document.querySelector('[data-action="train-stop"]'),w=document.querySelector('[data-action="generate-start"]'),h=document.querySelector('[data-action="generate-stop"]'),D=document.querySelector('[data-action="train-log"]'),G=document.querySelector('[data-action="generate-log"]'),P=document.querySelector('[data-action="dataset-list-file"]'),H=document.querySelector('[data-action="quota-category"]'),Q=document.querySelector('[data-action="quota-source"]'),W=document.querySelector('[data-action="quota-count"]'),M=document.querySelector('[data-action="shuffle-categories"]'),V=document.querySelector('[data-action="add-quota-row"]'),J=document.querySelector('[data-action="remove-quota-row"]'),g=document.querySelector('[data-action="quota-table-body"]'),S=document.querySelector('[data-action="output-catalog"]'),j=document.querySelector('[data-action="create-dataset-list"]'),O=document.querySelector('[data-action="config-status"]'),I=document.querySelector('[data-action="view-accession"]'),z=document.querySelector('[data-action="view-fasta-path"]'),_=document.querySelector('[data-action="view-render-mode"]'),b=document.querySelector('[data-action="view-output-path"]'),K=document.querySelector('[data-action="view-title"]'),d=document.querySelector('[data-action="view-generate-pdf"]'),q=document.querySelector('[data-action="view-open-pdf"]'),p=document.querySelector('[data-action="view-log"]');let i=null,o=null,f=null;function r(e){O&&(O.textContent=e)}function U(){return g?.querySelector('tr[data-action="quota-row"].is-selected')??null}function X(){g?.querySelectorAll('tr[data-action="quota-row"]').forEach(e=>{e.classList.remove("is-selected")})}function Y(e){if(!g)return;const a=document.createElement("tr");a.dataset.action="quota-row",a.dataset.category=e.category,a.dataset.source=e.source,a.dataset.count=String(e.count),a.innerHTML=`<td>${e.category}</td><td>${e.source}</td><td>${e.count}</td>`,a.addEventListener("click",()=>{const t=a.classList.contains("is-selected");X(),t||a.classList.add("is-selected")}),g.appendChild(a)}function Z(){const e=g?.querySelectorAll('tr[data-action="quota-row"]')??[];return Array.from(e).map(a=>({category:a.dataset.category??"",source:a.dataset.source??"",count:Number(a.dataset.count??"0")}))}function s(e,a){const t=e==="train"?D:G;t&&(t.textContent+=`${a}
`,t.scrollTop=t.scrollHeight)}function u(e){p&&(p.textContent+=`${e}
`,p.scrollTop=p.scrollHeight)}function E(e){q&&(q.disabled=!e)}function L(e,a){e==="train"?(y&&(y.disabled=a),m&&(m.disabled=!a)):(w&&(w.disabled=a),h&&(h.disabled=!a))}function k(){if(i&&(i.readyState===WebSocket.OPEN||i.readyState===WebSocket.CONNECTING))return i;const e=window.location.protocol==="https:"?"wss":"ws";return i=new WebSocket(`${e}://${window.location.host}/ws`),i.addEventListener("open",()=>{s("train","[web] Connected to backend."),s("generate","[web] Connected to backend."),u("[web] Connected to backend.")}),i.addEventListener("close",()=>{s("train","[web] Backend connection closed."),s("generate","[web] Backend connection closed."),u("[web] Backend connection closed."),o&&(L(o,!1),o=null)}),i.addEventListener("message",a=>{let t;try{t=JSON.parse(String(a.data))}catch{if(!o)return;s(o,`[web] Invalid backend message: ${String(a.data)}`);return}if(t.type==="create_dataset_result"){if(t.payload.logs?.length&&r(t.payload.logs.join(" | ")),t.payload.ok&&t.payload.output_catalog){S&&(S.value=t.payload.output_catalog),P&&(P.value=t.payload.output_catalog);const l=t.payload.selected_count??0;r(`Created dataset list with ${l} accessions at ${t.payload.output_catalog}.`)}else r(`Failed to create dataset list: ${t.payload.error??"unknown error"}`);return}if(t.type==="view_log"){u(t.message);return}if(t.type==="view_status"){t.status==="pending"||t.status==="done"||t.status==="error"?d&&(d.disabled=!1):d&&(d.disabled=!0);return}if(t.type==="view_result"){d&&(d.disabled=!1),t.payload.ok?(f=t.payload.file_url??null,E(!!f),t.payload.output_path&&b&&(b.value=t.payload.output_path),u(`[web] ${t.payload.status??"PDF generated."}`)):(f=null,E(!1),u(`[web] ERROR: ${t.payload.error??"Failed to generate PDF."}`));return}if(o){if(t.type==="log"){s(o,t.message);return}if(t.type==="status"){(t.status==="done"||t.status==="error"||t.status==="pending")&&(L(o,!1),o=null);return}if(t.type==="result"){const l=t.payload.exit_code;s(o,`[web] Run finished (exit code: ${l??"unknown"}).`)}}}),i}function A(e,a){const t=k();if(!a.trim()){s(e,"[web] Command is empty.");return}if(o){s(e,"[web] Another run is active. Stop it before starting a new one.");return}o=e,L(e,!0),s(e,`[web] Starting: ${a}`);const l=()=>{t.send(JSON.stringify({type:"start_run",command:a,cwd:"."}))};t.readyState===WebSocket.OPEN?l():t.addEventListener("open",l,{once:!0})}function B(e){if(!i||i.readyState!==WebSocket.OPEN){s(e,"[web] Backend is not connected.");return}if(o!==e){s(e,"[web] No active run in this tab.");return}i.send(JSON.stringify({type:"stop_run"})),s(e,"[web] Stop requested.")}y?.addEventListener("click",()=>A("train",F?.value??""));m?.addEventListener("click",()=>B("train"));w?.addEventListener("click",()=>A("generate",$?.value??""));h?.addEventListener("click",()=>B("generate"));V?.addEventListener("click",()=>{const e=H?.value?.trim()??"",a=Q?.value?.trim()??"",t=Number(W?.value??"0");if(!e||!a||!Number.isFinite(t)||t<=0){r("Please provide category, source catalog, and count > 0 before adding a quota row.");return}Y({category:e,source:a,count:Math.trunc(t)}),r(`Added quota row for ${e}.`)});function tt(){const e=k(),a={accession:I?.value?.trim()??"",fasta_path:z?.value?.trim()??"",render_mode:_?.value??"circular",title:K?.value?.trim()??"",output_path:b?.value?.trim()??""};if(!a.accession&&!a.fasta_path){u("[web] ERROR: Provide either accession or FASTA path.");return}p&&(p.textContent=""),f=null,E(!1),d&&(d.disabled=!0);const t=()=>{e.send(JSON.stringify({type:"view_generate_pdf",payload:a})),u("[web] View PDF generation requested.")};e.readyState===WebSocket.OPEN?t():e.addEventListener("open",t,{once:!0})}function et(){if(!f){u("[web] No generated PDF available yet.");return}window.open(f,"_blank","noopener")}_?.addEventListener("change",()=>{const e=_.value,a=b?.value?.trim()??"";b&&new Set(["generated/circular_genome.pdf","generated/linear_genome.pdf",""]).has(a)&&(b.value=e==="linear"?"generated/linear_genome.pdf":"generated/circular_genome.pdf")});d?.addEventListener("click",tt);q?.addEventListener("click",et);J?.addEventListener("click",()=>{const e=U();if(!e){r("Select a quota row to remove.");return}e.remove(),r("Removed selected quota row.")});j?.addEventListener("click",()=>{const e=k(),a=Z();if(!a.length){r("Add at least one category quota row before creating a dataset list.");return}const t=S?.value?.trim()??"";if(!t){r("Output catalog path is required.");return}const l=()=>{e.send(JSON.stringify({type:"create_dataset",payload:{category_quotas:a,selected_category_sources:a.map(n=>({category:n.category,source:n.source})),shuffle_within_category:!!M?.checked,output_catalog:t}})),r("Creating dataset list...")};e.readyState===WebSocket.OPEN?l():e.addEventListener("open",l,{once:!0})});
