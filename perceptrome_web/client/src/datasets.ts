import "./styles/index.css";
import { getMe, logout } from "./auth_api";
import { getDatasetPreview, listDatasets, type DatasetCatalogItem } from "./dataset_api";

function redirectToLogin(): void {
  const next = encodeURIComponent(window.location.pathname + window.location.search + window.location.hash);
  window.location.href = `/login.html?next=${next}`;
}

function redirectToChangePassword(): void {
  const next = encodeURIComponent(window.location.pathname + window.location.search + window.location.hash);
  window.location.href = `/change_password.html?next=${next}`;
}

function setMsg(message: string, type: "ok" | "error" | "plain" = "plain"): void {
  const el = document.getElementById("msg");
  if (!el) return;
  el.textContent = message;
  el.className = "msg" + (type === "plain" ? "" : ` ${type}`);
}

function useInRunUrl(dataset: DatasetCatalogItem): string {
  const params = new URLSearchParams({
    dataset: dataset.dataset_id,
    kind: "stream",
    config_path: "config/stream_config.yaml",
  });
  return `/index.html?${params.toString()}`;
}

function renderCards(rows: DatasetCatalogItem[]): void {
  const cards = document.getElementById("cards") as HTMLDivElement;
  if (!rows.length) {
    cards.innerHTML = '<div class="muted">No datasets match the current filter.</div>';
    return;
  }

  cards.innerHTML = rows
    .map((d) => {
      const splits = d.split_metadata.map((s) => `${s.name}:${s.count}`).join(" · ") || "n/a";
      return `
      <div class="panel-body">
        <div class="toolbar" style="justify-content: space-between; align-items: start;">
          <div class="stack">
            <strong>${d.dataset_id}</strong>
            <div class="muted">source=${d.source} | sequences=${d.sequence_count}</div>
            <div class="mono muted">hash=${d.last_updated_hash.slice(0, 12)}…</div>
            <div class="muted">splits: ${splits}</div>
            <div class="muted">tags: ${(d.tags || []).join(", ")}</div>
          </div>
          <div class="cluster">
            <button class="btn btn--secondary btn--sm" data-preview="${d.dataset_id}">Preview</button>
            <a class="btn btn--primary btn--sm" href="${useInRunUrl(d)}">Use in run</a>
          </div>
        </div>
      </div>`;
    })
    .join("");

  cards.querySelectorAll<HTMLButtonElement>("button[data-preview]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const datasetId = btn.dataset.preview;
      if (!datasetId) return;
      const previewEl = document.getElementById("preview") as HTMLElement;
      previewEl.textContent = "Loading preview…";
      try {
        const preview = await getDatasetPreview(datasetId, 25);
        previewEl.textContent = `${preview.dataset_id} (${preview.total_rows} rows)\n\n${preview.preview.join("\n")}`;
      } catch (err) {
        previewEl.textContent = `Preview error: ${String(err)}`;
      }
    });
  });
}

async function boot(): Promise<void> {
  const me = await getMe().catch(() => {
    redirectToLogin();
    throw new Error("Not authenticated");
  });
  if (me.must_change_password) {
    redirectToChangePassword();
    return;
  }

  const whoamiEl = document.getElementById("whoami");
  if (whoamiEl) whoamiEl.textContent = `${me.email} (${me.role})`;

  const logoutBtn = document.getElementById("logout-btn") as HTMLButtonElement;
  logoutBtn.addEventListener("click", async () => {
    try {
      await logout();
    } catch {
      // noop
    }
    window.location.href = "/login.html";
  });

  const searchEl = document.getElementById("search") as HTMLInputElement;
  const sourceEl = document.getElementById("source") as HTMLSelectElement;

  setMsg("Loading datasets...");
  const datasets = await listDatasets();
  const sources = Array.from(new Set(datasets.map((d) => d.source))).sort();
  sourceEl.innerHTML = '<option value="">All sources</option>' + sources.map((s) => `<option value="${s}">${s}</option>`).join("");

  const rerender = () => {
    const needle = searchEl.value.trim().toLowerCase();
    const source = sourceEl.value;
    const filtered = datasets.filter((d) => {
      if (source && d.source !== source) return false;
      if (!needle) return true;
      const hay = [d.dataset_id, d.source, ...(d.tags || [])].join(" ").toLowerCase();
      return hay.includes(needle);
    });
    renderCards(filtered);
    setMsg(`Showing ${filtered.length} / ${datasets.length} dataset(s).`, "ok");
  };

  searchEl.addEventListener("input", rerender);
  sourceEl.addEventListener("change", rerender);
  rerender();
}

if (document.readyState === "loading") {
  window.addEventListener("DOMContentLoaded", () => void boot());
} else {
  void boot();
}
