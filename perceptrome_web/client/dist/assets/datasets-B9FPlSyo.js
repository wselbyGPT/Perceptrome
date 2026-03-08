import{g,l as v}from"./auth_api-C3j5QkHe.js";async function w(e){const n=await fetch(e,{credentials:"include"});if(!n.ok){let t=`${n.status} ${n.statusText}`;try{const a=await n.json();t=a?.detail??a?.message??t}catch{}throw new Error(String(t))}return await n.json()}async function p(){return w("/api/datasets")}async function f(e,n=25){return w(`/api/datasets/${encodeURIComponent(e)}/preview?limit=${n}`)}function $(){const e=encodeURIComponent(window.location.pathname+window.location.search+window.location.hash);window.location.href=`/login.html?next=${e}`}function y(){const e=encodeURIComponent(window.location.pathname+window.location.search+window.location.hash);window.location.href=`/change_password.html?next=${e}`}function u(e,n="plain"){const t=document.getElementById("msg");t&&(t.textContent=e,t.className="msg"+(n==="plain"?"":` ${n}`))}function E(e){return`/index.html?${new URLSearchParams({dataset:e.dataset_id,kind:"stream",config_path:"config/stream_config.yaml"}).toString()}`}function L(e){const n=document.getElementById("cards");if(!e.length){n.innerHTML='<div class="muted">No datasets match the current filter.</div>';return}n.innerHTML=e.map(t=>{const a=t.split_metadata.map(o=>`${o.name}:${o.count}`).join(" · ")||"n/a";return`
      <div class="panel-body">
        <div class="toolbar" style="justify-content: space-between; align-items: start;">
          <div class="stack">
            <strong>${t.dataset_id}</strong>
            <div class="muted">source=${t.source} | sequences=${t.sequence_count}</div>
            <div class="mono muted">hash=${t.last_updated_hash.slice(0,12)}…</div>
            <div class="muted">splits: ${a}</div>
            <div class="muted">tags: ${(t.tags||[]).join(", ")}</div>
          </div>
          <div class="cluster">
            <button class="btn btn--secondary btn--sm" data-preview="${t.dataset_id}">Preview</button>
            <a class="btn btn--primary btn--sm" href="${E(t)}">Use in run</a>
          </div>
        </div>
      </div>`}).join(""),n.querySelectorAll("button[data-preview]").forEach(t=>{t.addEventListener("click",async()=>{const a=t.dataset.preview;if(!a)return;const o=document.getElementById("preview");o.textContent="Loading preview…";try{const s=await f(a,25);o.textContent=`${s.dataset_id} (${s.total_rows} rows)

${s.preview.join(`
`)}`}catch(s){o.textContent=`Preview error: ${String(s)}`}})})}async function m(){const e=await g().catch(()=>{throw $(),new Error("Not authenticated")});if(e.must_change_password){y();return}const n=document.getElementById("whoami");n&&(n.textContent=`${e.email} (${e.role})`),document.getElementById("logout-btn").addEventListener("click",async()=>{try{await v()}catch{}window.location.href="/login.html"});const a=document.getElementById("search"),o=document.getElementById("source");u("Loading datasets...");const s=await p(),h=Array.from(new Set(s.map(i=>i.source))).sort();o.innerHTML='<option value="">All sources</option>'+h.map(i=>`<option value="${i}">${i}</option>`).join("");const c=()=>{const i=a.value.trim().toLowerCase(),d=o.value,l=s.filter(r=>d&&r.source!==d?!1:i?[r.dataset_id,r.source,...r.tags||[]].join(" ").toLowerCase().includes(i):!0);L(l),u(`Showing ${l.length} / ${s.length} dataset(s).`,"ok")};a.addEventListener("input",c),o.addEventListener("change",c),c()}document.readyState==="loading"?window.addEventListener("DOMContentLoaded",()=>{m()}):m();
