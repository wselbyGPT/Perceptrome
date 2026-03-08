import { mustEl } from "./dom";

export class MetricsPanel {
  private metricsEl = mustEl<HTMLElement>("metrics");

  pushMetric(name: string, value: unknown): void {
    const line = `${name}: ${String(value ?? "")}`;
    this.metricsEl.textContent = this.metricsEl.textContent ? `${this.metricsEl.textContent}\n${line}` : line;
  }
}
