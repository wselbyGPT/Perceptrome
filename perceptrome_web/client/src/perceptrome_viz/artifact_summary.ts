import { type RunRecord } from "../run_api";
import { asPrettyText, mustEl } from "./dom";

export class ArtifactSummary {
  private resultsEl = mustEl<HTMLElement>("results");
  private generatedEl = mustEl<HTMLElement>("generated-sequences");
  private validationEl = mustEl<HTMLElement>("validation-results");
  private checkpointsEl = mustEl<HTMLElement>("checkpoints");

  renderResults(value: unknown): void {
    this.resultsEl.textContent = asPrettyText(value);
    const payload = value && typeof value === "object" ? (value as Record<string, unknown>) : {};
    this.generatedEl.textContent = asPrettyText(payload.generated_sequences ?? payload.generated ?? []);
    this.validationEl.textContent = asPrettyText(payload.validation_results ?? payload.validation ?? {});
  }

  renderArtifacts(run: RunRecord): void {
    const links = run.artifacts.map((a) => `<a href="${a.download_url}">${a.label ?? a.path}</a>`).join("\n");
    this.checkpointsEl.innerHTML = links || "No artifacts";
  }

  pushArtifact(path: string, downloadUrl?: string): void {
    this.checkpointsEl.innerHTML += `${downloadUrl ? `<a href="${downloadUrl}">${path || downloadUrl}</a>` : path}<br/>`;
  }
}
