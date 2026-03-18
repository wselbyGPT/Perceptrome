export type JsonRecord = Record<string, unknown>;

export function mustEl<T extends Element>(root: ParentNode, id: string): T {
  const el = root.querySelector<T>(`#${id}`);
  if (!el) throw new Error(`Missing required element #${id}`);
  return el;
}

export function asPrettyText(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function nowStamp(): string {
  return new Date().toLocaleTimeString();
}
