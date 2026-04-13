import re

from pydantic import BaseModel, Field, field_validator


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


class DatasetCreateRequest(BaseModel):
    dataset_id: str = Field(..., min_length=1, max_length=120)
    accessions: list[str] = Field(..., min_length=1)

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, v: str) -> str:
        v = v.strip()
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_\-]*$", v):
            raise ValueError("dataset_id must be alphanumeric (underscores and hyphens allowed)")
        return v

    @field_validator("accessions")
    @classmethod
    def validate_accessions(cls, v: list[str]) -> list[str]:
        cleaned = [a.strip() for a in v if a.strip()]
        if not cleaned:
            raise ValueError("At least one accession is required")
        return cleaned
