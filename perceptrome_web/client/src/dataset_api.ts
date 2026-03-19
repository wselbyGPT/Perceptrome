export type { DatasetCatalogItem, DatasetDetail, DatasetPreview, DatasetSplit } from "./features/datasets/api";
import { datasetsApi } from "./features/datasets/api";

export const listDatasets = () => datasetsApi.list();
export const getDataset = (datasetId: string) => datasetsApi.detail(datasetId);
export const getDatasetPreview = (datasetId: string, limit = 25) => datasetsApi.preview(datasetId, limit);
