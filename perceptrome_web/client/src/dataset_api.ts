export type DatasetSplit = {
  name: string;
  count: number;
};

export type DatasetCatalogItem = {
  dataset_id: string;
  source: string;
  sequence_count: number;
  split_metadata: DatasetSplit[];
  tags: string[];
  last_updated_hash: string;
};

export type DatasetDetail = DatasetCatalogItem & {
  manifest_path: string;
};

export type DatasetPreview = {
  dataset_id: string;
  source: string;
  preview: string[];
  total_rows: number;
};

async function fetchJson<T>(url: string): Promise<T> {
  const resp = await fetch(url, { credentials: "include" });
  if (!resp.ok) {
    let detail = `${resp.status} ${resp.statusText}`;
    try {
      const body = await resp.json();
      detail = body?.detail ?? body?.message ?? detail;
    } catch {
      // noop
    }
    throw new Error(String(detail));
  }
  return (await resp.json()) as T;
}

export async function listDatasets(): Promise<DatasetCatalogItem[]> {
  return fetchJson<DatasetCatalogItem[]>("/api/datasets");
}

export async function getDataset(datasetId: string): Promise<DatasetDetail> {
  return fetchJson<DatasetDetail>(`/api/datasets/${encodeURIComponent(datasetId)}`);
}

export async function getDatasetPreview(datasetId: string, limit = 25): Promise<DatasetPreview> {
  return fetchJson<DatasetPreview>(`/api/datasets/${encodeURIComponent(datasetId)}/preview?limit=${limit}`);
}
