from pydantic import BaseModel, Field


class DatasetSplitOut(BaseModel):
    name: str
    count: int


class DatasetCatalogItemOut(BaseModel):
    dataset_id: str
    source: str
    sequence_count: int
    split_metadata: list[DatasetSplitOut] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    last_updated_hash: str


class DatasetDetailOut(DatasetCatalogItemOut):
    manifest_path: str


class DatasetPreviewOut(BaseModel):
    dataset_id: str
    source: str
    preview: list[str] = Field(default_factory=list)
    total_rows: int
