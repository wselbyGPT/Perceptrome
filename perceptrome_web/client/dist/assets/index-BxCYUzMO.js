(function(){const a=document.createElement("link").relList;if(a&&a.supports&&a.supports("modulepreload"))return;for(const o of document.querySelectorAll('link[rel="modulepreload"]'))n(o);new MutationObserver(o=>{for(const l of o)if(l.type==="childList")for(const O of l.addedNodes)O.tagName==="LINK"&&O.rel==="modulepreload"&&n(O)}).observe(document,{childList:!0,subtree:!0});function t(o){const l={};return o.integrity&&(l.integrity=o.integrity),o.referrerPolicy&&(l.referrerPolicy=o.referrerPolicy),o.crossOrigin==="use-credentials"?l.credentials="include":o.crossOrigin==="anonymous"?l.credentials="omit":l.credentials="same-origin",l}function n(o){if(o.ep)return;o.ep=!0;const l=t(o);fetch(o.href,l)}})();const de=document.querySelector("#app");de.innerHTML=`
  <div class="app-root">
    <header class="app-header">
      <div class="app-title">Perceptrome</div>
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
                  <input class="field-input" type="text" data-action="project-dir" value="." />
                </label>

                <label class="field">
                  <span class="field-label">stream_config.yaml:</span>
                  <input class="field-input" type="text" data-action="stream-config" value="stream_config.yaml" />
                </label>

                <label class="field">
                  <span class="field-label">Dataset list file:</span>
                  <input class="field-input" type="text" data-action="dataset-list-file" value="config/plasmids_10.txt" />
                </label>

                <label class="field">
                  <span class="field-label">Epochs:</span>
                  <input class="field-input" type="number" data-action="epochs" value="10" />
                </label>

                <label class="field">
                  <span class="field-label">Batch size:</span>
                  <input class="field-input" type="number" data-action="batch-size" value="256" />
                </label>

                <label class="field">
                  <span class="field-label">Learning rate:</span>
                  <input class="field-input" type="number" step="0.000001" data-action="learning-rate" value="0.001" />
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
                <select class="field-input" data-action="train-model-type">
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
              <button class="btn btn-secondary" type="button" data-action="train-help">Help</button>
              <button class="btn btn-secondary" type="button" data-action="train-build-command">Build command</button>
              <button class="btn" type="button" data-action="train-start">Start</button>
              <button class="btn btn-danger" type="button" data-action="train-stop" disabled>Stop</button>
            </div>

            <div class="progress-bar" data-action="train-progress">
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
                <select class="field-input" data-action="generate-model">
                  <option>model/checkpoints/latest.pt</option>
                </select>
              </label>

              <label class="field">
                <span class="field-label">Length (bp):</span>
                <input class="field-input" type="number" data-action="generate-length" value="10000" />
              </label>

              <label class="field">
                <span class="field-label">Output FASTA:</span>
                <input class="field-input" type="text" data-action="generate-output" value="generated/novel_plasmid.fasta" />
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
              <button class="btn btn-secondary" type="button" data-action="generate-help">Help</button>
              <button class="btn btn-secondary" type="button" data-action="generate-refresh-models">Refresh models</button>
              <button class="btn btn-secondary" type="button" data-action="generate-build-command">Build command</button>
              <button class="btn" type="button" data-action="generate-start">Start</button>
              <button class="btn btn-danger" type="button" data-action="generate-stop" disabled>Stop</button>
            </div>

            <div class="progress-bar" data-action="generate-progress">
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
              <button class="btn btn-secondary" type="button" data-action="clear-history">Clear history</button>
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
                <tbody data-action="history-table-body">
                  <!-- Empty, matches Qt screenshot -->
                </tbody>
              </table>
            </div>
          </section>
        </section>
      </main>
    </div>
  </div>
`;const j=Array.from(document.querySelectorAll(".tab")),ue=Array.from(document.querySelectorAll(".tab-panel"));function Q(e){j.forEach(a=>{const t=a.dataset.tab===e;a.classList.toggle("tab--active",t)}),ue.forEach(a=>{const t=a.dataset.tabPanel===e;a.classList.toggle("tab-panel--active",t)})}j.forEach(e=>{e.addEventListener("click",()=>{const a=e.dataset.tab;Q(a)})});const D=document.querySelector('[data-action="go-train"]');D&&D.addEventListener("click",()=>Q("train"));const E=document.querySelector('[data-action="train-command"]'),_=document.querySelector('[data-action="generate-command"]'),F=document.querySelector('[data-action="project-dir"]'),H=document.querySelector('[data-action="stream-config"]'),I=document.querySelector('[data-action="epochs"]'),J=document.querySelector('[data-action="batch-size"]'),W=document.querySelector('[data-action="learning-rate"]'),z=document.querySelector('[data-action="train-model-type"]'),pe=document.querySelector('[data-action="train-help"]'),fe=document.querySelector('[data-action="train-build-command"]'),T=document.querySelector('[data-action="train-start"]'),B=document.querySelector('[data-action="train-stop"]'),be=document.querySelector('[data-action="generate-help"]'),ge=document.querySelector('[data-action="generate-refresh-models"]'),me=document.querySelector('[data-action="generate-build-command"]'),y=document.querySelector('[data-action="generate-model"]'),V=document.querySelector('[data-action="generate-length"]'),K=document.querySelector('[data-action="generate-output"]'),$=document.querySelector('[data-action="generate-start"]'),x=document.querySelector('[data-action="generate-stop"]'),ye=document.querySelector('[data-action="train-log"]'),ve=document.querySelector('[data-action="generate-log"]'),he=document.querySelector('[data-action="train-progress"] .progress-bar__value'),we=document.querySelector('[data-action="train-progress"] .progress-bar__label'),Se=document.querySelector('[data-action="generate-progress"] .progress-bar__value'),Ee=document.querySelector('[data-action="generate-progress"] .progress-bar__label'),L=document.querySelector('[data-action="dataset-list-file"]'),U=document.querySelector('[data-action="quota-category"]'),Y=document.querySelector('[data-action="quota-source"]'),Z=document.querySelector('[data-action="quota-count"]'),X=document.querySelector('[data-action="shuffle-categories"]'),_e=document.querySelector('[data-action="add-quota-row"]'),qe=document.querySelector('[data-action="remove-quota-row"]'),m=document.querySelector('[data-action="quota-table-body"]'),k=document.querySelector('[data-action="output-catalog"]'),Le=document.querySelector('[data-action="create-dataset-list"]'),G=document.querySelector('[data-action="config-status"]'),ee=document.querySelector('[data-action="view-accession"]'),te=document.querySelector('[data-action="view-fasta-path"]'),C=document.querySelector('[data-action="view-render-mode"]'),g=document.querySelector('[data-action="view-output-path"]'),ae=document.querySelector('[data-action="view-title"]'),p=document.querySelector('[data-action="view-generate-pdf"]'),N=document.querySelector('[data-action="view-open-pdf"]'),v=document.querySelector('[data-action="view-log"]'),q=document.querySelector('[data-action="history-table-body"]'),ke=document.querySelector('[data-action="clear-history"]'),ne="perceptrome.web.state.v1",M={datasetListFile:L,quotaCategory:U,quotaSource:Y,quotaCount:Z,shuffleCategories:X,outputCatalog:k,projectDir:F,streamConfig:H,epochs:I,batchSize:J,learningRate:W,trainModelType:z,generateModel:y,generateLength:V,generateOutput:K,trainCommand:E,generateCommand:_,viewAccession:ee,viewFastaPath:te,viewRenderMode:C,viewOutputPath:g,viewTitle:ae},h=[];let d=null,i=null,w=null;function Ce(e){return typeof e!="number"||Number.isNaN(e)?0:Math.max(0,Math.min(1,e))}function S(e,a){const t=Ce(a),n=`${Math.round(t*100)}%`,o=e==="train"?he:Se,l=e==="train"?we:Ee;o&&(o.style.width=n),l&&(l.textContent=n)}function u(e){return e?/^[a-zA-Z0-9_./:-]+$/.test(e)?e:`'${e.replace(/'/g,"'''")}'`:"''"}function Pe(){const e=H?.value?.trim()||"stream_config.yaml",a=L?.value?.trim()||"config/plasmids_10.txt",t=z?.value||"mlp",n=I?.value?.trim()||"10",o=J?.value?.trim()||"256",l=W?.value?.trim()||"0.001";return["perceptrome","--config",u(e),"stream","--catalog",u(a),"--model-type",u(t),"--steps-per-plasmid",u(n),"--batch-size",u(o),"--learning-rate",u(l)].join(" ")}function Oe(){const e=H?.value?.trim()||"stream_config.yaml",a=y?.value?.trim()||"model/checkpoints/latest.pt",t=V?.value?.trim()||"10000",n=K?.value?.trim()||"generated/novel_plasmid.fasta";return["perceptrome","--config",u(e),"generate-plasmid","--checkpoint",u(a),"--length-bp",u(t),"--output",u(n)].join(" ")}function oe(){const e=P(),a=F?.value?.trim()||".",t=()=>e.send(JSON.stringify({type:"list_models",cwd:a}));e.readyState===WebSocket.OPEN?t():e.addEventListener("open",t,{once:!0})}function Te(e){y&&(y.innerHTML="",e.forEach(a=>{const t=document.createElement("option");t.value=a,t.textContent=a,y.appendChild(t)}),e.length>0&&(y.value=e[0]))}function r(e){G&&(G.textContent=e)}function c(){const e={};Object.entries(M).forEach(([t,n])=>{if(n){if(n instanceof HTMLInputElement&&n.type==="checkbox"){e[t]=n.checked;return}e[t]=n.value}});const a={config:e,categoryQuotas:ie(),history:h};localStorage.setItem(ne,JSON.stringify(a))}function se(e){if(!q)return;const a=document.createElement("tr");a.innerHTML=`<td>${e.time}</td><td>${e.action}</td><td>${e.details}</td>`,q.appendChild(a)}function f(e,a){const t={time:new Date().toLocaleString(),action:e,details:a};h.push(t),se(t),c()}function Be(){const e=localStorage.getItem(ne);if(!e){r("No saved config found.");return}try{const a=JSON.parse(e),t=a.config??{};Object.entries(M).forEach(([n,o])=>{if(!o)return;const l=t[n];l!==void 0&&(o instanceof HTMLInputElement&&o.type==="checkbox"?o.checked=!!l:o.value=String(l))}),m&&(m.innerHTML="",(a.categoryQuotas??[]).forEach(n=>le(n))),q&&(q.innerHTML=""),h.splice(0,h.length,...a.history??[]),h.forEach(n=>se(n)),r("Loaded saved config.")}catch{r("Saved config could not be loaded.")}}function $e(){return m?.querySelector('tr[data-action="quota-row"].is-selected')??null}function xe(){m?.querySelectorAll('tr[data-action="quota-row"]').forEach(e=>{e.classList.remove("is-selected")})}function le(e){if(!m)return;const a=document.createElement("tr");a.dataset.action="quota-row",a.dataset.category=e.category,a.dataset.source=e.source,a.dataset.count=String(e.count),a.innerHTML=`<td>${e.category}</td><td>${e.source}</td><td>${e.count}</td>`,a.addEventListener("click",()=>{const t=a.classList.contains("is-selected");xe(),t||a.classList.add("is-selected")}),m.appendChild(a),c()}function ie(){const e=m?.querySelectorAll('tr[data-action="quota-row"]')??[];return Array.from(e).map(a=>({category:a.dataset.category??"",source:a.dataset.source??"",count:Number(a.dataset.count??"0")}))}function s(e,a){const t=e==="train"?ye:ve;t&&(t.textContent+=`${a}
`,t.scrollTop=t.scrollHeight)}function b(e){v&&(v.textContent+=`${e}
`,v.scrollTop=v.scrollHeight)}function R(e){N&&(N.disabled=!e)}function A(e,a){e==="train"?(T&&(T.disabled=a),B&&(B.disabled=!a)):($&&($.disabled=a),x&&(x.disabled=!a))}function P(){if(d&&(d.readyState===WebSocket.OPEN||d.readyState===WebSocket.CONNECTING))return d;const e=window.location.protocol==="https:"?"wss":"ws";return d=new WebSocket(`${e}://${window.location.host}/ws`),d.addEventListener("open",()=>{s("train","[web] Connected to backend."),s("generate","[web] Connected to backend."),b("[web] Connected to backend.")}),d.addEventListener("close",()=>{s("train","[web] Backend connection closed."),s("generate","[web] Backend connection closed."),b("[web] Backend connection closed."),i&&(A(i,!1),S(i,0),i=null)}),d.addEventListener("message",a=>{let t;try{t=JSON.parse(String(a.data))}catch{if(!i)return;s(i,`[web] Invalid backend message: ${String(a.data)}`);return}if(t.type==="create_dataset_result"){if(t.payload.logs?.length&&r(t.payload.logs.join(" | ")),t.payload.ok&&t.payload.output_catalog){k&&(k.value=t.payload.output_catalog),L&&(L.value=t.payload.output_catalog);const n=t.payload.selected_count??0;r(`Created dataset list with ${n} accessions at ${t.payload.output_catalog}.`),f("dataset-create",`Created ${t.payload.output_catalog} (${n} accessions)`)}else r(`Failed to create dataset list: ${t.payload.error??"unknown error"}`),f("failure",`Dataset create failed: ${t.payload.error??"unknown error"}`);c();return}if(t.type==="view_log"){b(t.message);return}if(t.type==="view_status"){t.status==="pending"||t.status==="done"||t.status==="error"?p&&(p.disabled=!1):p&&(p.disabled=!0);return}if(t.type==="view_result"){p&&(p.disabled=!1),t.payload.ok?(w=t.payload.file_url??null,R(!!w),t.payload.output_path&&g&&(g.value=t.payload.output_path),b(`[web] ${t.payload.status??"PDF generated."}`),f("view-generate",`Generated ${t.payload.output_path??"PDF"}`)):(w=null,R(!1),b(`[web] ERROR: ${t.payload.error??"Failed to generate PDF."}`),f("failure",`View generate failed: ${t.payload.error??"unknown error"}`)),c();return}if(t.type==="model_list"){if(!t.payload.ok){s("generate",`[web] Failed to list models: ${t.payload.error??"unknown error"}`);return}Te(t.payload.checkpoints),s("generate",`[web] Loaded ${t.payload.checkpoints.length} checkpoint(s).`),t.payload.checkpoints.length===0&&s("generate","[web] No checkpoints found. Train first or check project directory.");return}if(i){if(t.type==="log"){s(i,t.message);return}if(t.type==="status"){S(i,t.progress),(t.status==="done"||t.status==="error"||t.status==="pending")&&(t.status==="done"&&f("success",`${i}: run completed`),t.status==="error"&&f("failure",`${i}: run failed`),A(i,!1),t.status==="pending"&&S(i,0),i=null);return}if(t.type==="result"){const n=t.payload.exit_code,o=t.payload.output_paths??[];s(i,`[web] RESULT ${JSON.stringify({exit_code:n??null,output_paths:o},null,2)}`),s(i,`[web] Run finished (exit code: ${n??"unknown"}).`);return}}}),d}function re(e,a){const t=P();if(!a.trim()){s(e,"[web] Command is empty.");return}if(i){s(e,"[web] Another run is active. Stop it before starting a new one.");return}i=e,A(e,!0),S(e,0),s(e,`[web] Starting: ${a}`),f("start",`${e}: ${a}`);const n=()=>{t.send(JSON.stringify({type:"start_run",command:a,cwd:F?.value?.trim()||"."}))};t.readyState===WebSocket.OPEN?n():t.addEventListener("open",n,{once:!0})}function ce(e){if(!d||d.readyState!==WebSocket.OPEN){s(e,"[web] Backend is not connected.");return}if(i!==e){s(e,"[web] No active run in this tab.");return}d.send(JSON.stringify({type:"stop_run"})),s(e,"[web] Stop requested."),f("stop",`${e}: stop requested`)}pe?.addEventListener("click",()=>{E&&(E.value="perceptrome --help && perceptrome stream --help",s("train","[web] Filled train help command."),c())});fe?.addEventListener("click",()=>{E&&(E.value=Pe(),s("train","[web] Built train command from config fields."),c())});be?.addEventListener("click",()=>{_&&(_.value="perceptrome --help && perceptrome generate-plasmid --help",s("generate","[web] Filled generate help command."),c())});me?.addEventListener("click",()=>{_&&(_.value=Oe(),s("generate","[web] Built generate command from config fields."),c())});ge?.addEventListener("click",()=>{s("generate","[web] Refreshing checkpoints from backend..."),oe()});T?.addEventListener("click",()=>re("train",E?.value??""));B?.addEventListener("click",()=>ce("train"));$?.addEventListener("click",()=>re("generate",_?.value??""));x?.addEventListener("click",()=>ce("generate"));_e?.addEventListener("click",()=>{const e=U?.value?.trim()??"",a=Y?.value?.trim()??"",t=Number(Z?.value??"0");if(!e||!a||!Number.isFinite(t)||t<=0){r("Please provide category, source catalog, and count > 0 before adding a quota row.");return}le({category:e,source:a,count:Math.trunc(t)}),r(`Added quota row for ${e}.`),c()});function Ne(){const e=P(),a={accession:ee?.value?.trim()??"",fasta_path:te?.value?.trim()??"",render_mode:C?.value??"circular",title:ae?.value?.trim()??"",output_path:g?.value?.trim()??""};if(!a.accession&&!a.fasta_path){b("[web] ERROR: Provide either accession or FASTA path.");return}v&&(v.textContent=""),w=null,R(!1),p&&(p.disabled=!0);const t=()=>{e.send(JSON.stringify({type:"view_generate_pdf",payload:a})),b("[web] View PDF generation requested.")};e.readyState===WebSocket.OPEN?t():e.addEventListener("open",t,{once:!0})}function Re(){if(!w){b("[web] No generated PDF available yet.");return}window.open(w,"_blank","noopener")}C?.addEventListener("change",()=>{const e=C.value,a=g?.value?.trim()??"";g&&new Set(["generated/circular_genome.pdf","generated/linear_genome.pdf",""]).has(a)&&(g.value=e==="linear"?"generated/linear_genome.pdf":"generated/circular_genome.pdf"),c()});p?.addEventListener("click",Ne);N?.addEventListener("click",Re);qe?.addEventListener("click",()=>{const e=$e();if(!e){r("Select a quota row to remove.");return}e.remove(),r("Removed selected quota row."),c()});Le?.addEventListener("click",()=>{const e=P(),a=ie();if(!a.length){r("Add at least one category quota row before creating a dataset list.");return}const t=k?.value?.trim()??"";if(!t){r("Output catalog path is required.");return}const n=()=>{e.send(JSON.stringify({type:"create_dataset",payload:{category_quotas:a,selected_category_sources:a.map(o=>({category:o.category,source:o.source})),shuffle_within_category:!!X?.checked,output_catalog:t}})),r("Creating dataset list...")};e.readyState===WebSocket.OPEN?n():e.addEventListener("open",n,{once:!0})});ke?.addEventListener("click",()=>{h.length=0,q&&(q.innerHTML=""),c(),r("History cleared.")});Object.values(M).forEach(e=>{if(!e)return;const a=e instanceof HTMLInputElement&&e.type==="checkbox"?"change":"input";e.addEventListener(a,()=>c())});S("train",0);S("generate",0);Be();oe();
