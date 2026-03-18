import { mustEl, nowStamp } from "./dom";

export class ActiveRunTimeline {
  private logsEl: HTMLElement;
  private timelineEl: HTMLElement;

  constructor(root: ParentNode) {
    this.logsEl = mustEl<HTMLElement>(root, "logs");
    this.timelineEl = mustEl<HTMLElement>(root, "active-run-timeline");
  }

  appendLog(line: string, opts?: { kind?: "info" | "warn" | "error" | "raw"; noStamp?: boolean }): void {
    const kind = opts?.kind ?? "info";
    const prefix = opts?.noStamp ? "" : `[${nowStamp()}] `;
    const tag = kind === "error" ? "[ERR] " : kind === "warn" ? "[WRN] " : "";
    const text = `${prefix}${tag}${line}`;
    this.logsEl.textContent = this.logsEl.textContent ? `${this.logsEl.textContent}\n${text}` : text;
    this.logsEl.scrollTop = this.logsEl.scrollHeight;
  }

  pushTimelineEvent(label: string): void {
    const row = document.createElement("div");
    row.className = "timeline-row";
    row.textContent = `[${nowStamp()}] ${label}`;
    this.timelineEl.prepend(row);
  }
}
