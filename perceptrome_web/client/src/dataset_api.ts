import { apiRequest } from "./lib/api-client";

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

export async function listDatasets(): Promise<DatasetCatalogItem[]> {
  return apiRequest<DatasetCatalogItem[]>("/api/datasets");
}

export async function getDataset(datasetId: string): Promise<DatasetDetail> {
  return apiRequest<DatasetDetail>(`/api/datasets/${encodeURIComponent(datasetId)}`);
}

export async function getDatasetPreview(datasetId: string, limit = 25): Promise<DatasetPreview> {
  return apiRequest<DatasetPreview>(`/api/datasets/${encodeURIComponent(datasetId)}/preview?limit=${limit}`);
}
