(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const t of document.querySelectorAll('link[rel="modulepreload"]'))i(t);new MutationObserver(t=>{for(const a of t)if(a.type==="childList")for(const n of a.addedNodes)n.tagName==="LINK"&&n.rel==="modulepreload"&&i(n)}).observe(document,{childList:!0,subtree:!0});function s(t){const a={};return t.integrity&&(a.integrity=t.integrity),t.referrerPolicy&&(a.referrerPolicy=t.referrerPolicy),t.crossOrigin==="use-credentials"?a.credentials="include":t.crossOrigin==="anonymous"?a.credentials="omit":a.credentials="same-origin",a}function i(t){if(t.ep)return;t.ep=!0;const a=s(t);fetch(t.href,a)}})();const r=document.querySelector("#app");r.innerHTML=`
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
                  <input class="field-input" type="text" value="config/plasmids_10.txt" />
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
                  <select class="field-input">
                    <option>Plasmids</option>
                    <option>Bacterial genomes</option>
                    <option>Custom</option>
                  </select>
                </label>

                <label class="field">
                  <span class="field-label">Count:</span>
                  <input class="field-input" type="number" value="100" />
                </label>
              </div>

              <label class="checkbox">
                <input type="checkbox" checked />
                <span>Shuffle inside each category</span>
              </label>

              <div class="button-row">
                <button class="btn btn-secondary" type="button">Add category quota</button>
                <button class="btn btn-secondary" type="button">Remove selected</button>
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
                  <tbody>
                    <!-- Empty state, matches Qt view -->
                  </tbody>
                </table>
              </div>

              <div class="form-grid">
                <label class="field">
                  <span class="field-label">Output catalog:</span>
                  <input class="field-input" type="text" value="config/custom_dataset.txt" />
                </label>
              </div>

              <div class="button-row">
                <button class="btn" type="button">Create dataset list</button>
              </div>

              <div class="button-row button-row--end">
                <button class="btn btn-secondary" type="button">Save config</button>
                <button class="btn" type="button" data-action="go-train">Go to Train tab</button>
              </div>

              <p class="panel-footer-text">Loaded saved config (if any).</p>
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
                  value=".yaml stream --catalog config/plasmids_10.txt --model-type mlp --steps-per-plasmid 10 --batch-size 256"
                />
              </label>
            </div>

            <div class="button-row">
              <button class="btn btn-secondary" type="button">Help</button>
              <button class="btn btn-secondary" type="button">Build command</button>
              <button class="btn" type="button">Start</button>
              <button class="btn btn-danger" type="button">Stop</button>
            </div>

            <div class="progress-bar">
              <div class="progress-bar__track">
                <div class="progress-bar__value" style="width: 0%;"></div>
              </div>
              <div class="progress-bar__label">0%</div>
            </div>

            <div class="log-area">
              <div class="log-area__label">Live training output will appear here...</div>
              <pre class="log-area__body"></pre>
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
                  value="--config stream_config.yaml generate-plasmid --length-bp 10000 --output generated/novel_plasmid.fasta"
                />
              </label>
            </div>

            <div class="button-row">
              <button class="btn btn-secondary" type="button">Help</button>
              <button class="btn btn-secondary" type="button">Refresh models</button>
              <button class="btn btn-secondary" type="button">Build command</button>
              <button class="btn" type="button">Start</button>
              <button class="btn btn-danger" type="button">Stop</button>
            </div>

            <div class="progress-bar">
              <div class="progress-bar__track">
                <div class="progress-bar__value" style="width: 0%;"></div>
              </div>
              <div class="progress-bar__label">0%</div>
            </div>

            <div class="log-area">
              <div class="log-area__label">Live generate output will appear here...</div>
              <pre class="log-area__body"></pre>
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
`;const c=Array.from(document.querySelectorAll(".tab")),p=Array.from(document.querySelectorAll(".tab-panel"));function d(l){c.forEach(e=>{const s=e.dataset.tab===l;e.classList.toggle("tab--active",s)}),p.forEach(e=>{const s=e.dataset.tabPanel===l;e.classList.toggle("tab-panel--active",s)})}c.forEach(l=>{l.addEventListener("click",()=>{const e=l.dataset.tab;d(e)})});const o=document.querySelector('[data-action="go-train"]');o&&o.addEventListener("click",()=>d("train"));
