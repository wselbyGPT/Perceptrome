import { asPrettyText, mustEl } from "./dom";
import { listActiveRuns, listFailedRuns, type RunRecord, type RunSummary } from "../run_api";

export class RunQueueBoard {
  private statusEl: HTMLElement;
  private summaryEl: HTMLElement;
  private activeEl: HTMLElement;
  private failedEl: HTMLElement;

  constructor(root: ParentNode) {
    this.statusEl = mustEl<HTMLElement>(root, "status");
    this.summaryEl = mustEl<HTMLElement>(root, "run-summary-cards");
    this.activeEl = mustEl<HTMLElement>(root, "active-run-board");
    this.failedEl = mustEl<HTMLElement>(root, "failed-run-board");
  }

  setStatus(text: string, progress?: number | null): void {
    if (typeof progress === "number" && Number.isFinite(progress)) {
      const pct = Math.max(0, Math.min(100, Math.round(progress * 100)));
      this.statusEl.textContent = `${text} (${pct}%)`;
      return;
    }
    this.statusEl.textContent = text;
  }

  renderSummary(summary: RunSummary): void {
    const cards = [["Queued", summary.queued], ["Running", summary.running], ["Failed", summary.failed], ["Completed", summary.completed]];
    this.summaryEl.innerHTML = cards.map(([label, count]) => `<article class="mini-card"><h3>${label}</h3><p>${count}</p></article>`).join("");
  }

  renderRuns(target: HTMLElement, runs: RunRecord[], emptyLabel: string, onInspect: (runId: string) => void): void {
    target.innerHTML = "";
    if (runs.length === 0) {
      target.textContent = emptyLabel;
      return;
    }
    for (const run of runs) {
      const item = document.createElement("div");
      item.className = "stack run-row";
      item.innerHTML = `<strong>${run.run_id}</strong> <span>[${run.kind}] ${run.state}</span>`;
      const btn = document.createElement("button");
      btn.className = "btn btn--secondary btn--sm";
      btn.type = "button";
      btn.textContent = "Drill down";
      btn.addEventListener("click", () => onInspect(run.run_id));
      item.appendChild(btn);
      target.appendChild(item);
    }
  }

  async refresh(onInspect: (runId: string) => void): Promise<{ activeRuns: RunRecord[]; failedRuns: RunRecord[] }> {
    const [activeResp, failedResp] = await Promise.all([listActiveRuns(12), listFailedRuns(12)]);
    this.renderRuns(this.activeEl, activeResp.runs, "No active runs.", onInspect);
    this.renderRuns(this.failedEl, failedResp.runs, "No recent failures.", onInspect);
    return { activeRuns: activeResp.runs, failedRuns: failedResp.runs };
  }

  showBoardError(err: unknown): void {
    this.activeEl.textContent = `Failed to load active runs: ${asPrettyText(err)}`;
  }
}
