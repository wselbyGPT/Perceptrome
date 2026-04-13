from pathlib import Path
import re

from fastapi import HTTPException

from ..schemas import DatasetCatalogItemOut, DatasetCreateRequest, DatasetDetailOut, DatasetPreviewOut, DatasetSplitOut
from .run_service import sha256_file


def dataset_config_dir() -> Path:
    return Path(__file__).resolve().parents[3] / 'config'


def count_dataset_sequences(path: Path) -> int:
    count = 0
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def dataset_split_metadata(dataset_id: str, sequence_count: int) -> list[DatasetSplitOut]:
    if sequence_count <= 0:
        return []
    if '_' in dataset_id:
        prefix = dataset_id.split('_', 1)[0]
        siblings = sorted(dataset_config_dir().glob(f'{prefix}_*.txt'))
        if len(siblings) >= 2:
            return [DatasetSplitOut(name='catalog_group', count=len(siblings))]
    train_count = int(round(sequence_count * 0.8))
    val_count = int(round(sequence_count * 0.1))
    test_count = max(sequence_count - train_count - val_count, 0)
    return [DatasetSplitOut(name='train', count=train_count), DatasetSplitOut(name='validation', count=val_count), DatasetSplitOut(name='test', count=test_count)]


def build_dataset_catalog_item(path: Path) -> DatasetCatalogItemOut:
    dataset_id = path.stem
    source = dataset_id.split('_', 1)[0]
    sequence_count = count_dataset_sequences(path)
    return DatasetCatalogItemOut(dataset_id=dataset_id, source=source, sequence_count=sequence_count, split_metadata=dataset_split_metadata(dataset_id, sequence_count), tags=sorted({source, f'size:{sequence_count}', 'manifest', 'config'}), last_updated_hash=sha256_file(path))


def load_dataset_catalog() -> list[DatasetCatalogItemOut]:
    cfg = dataset_config_dir()
    if not cfg.exists() or not cfg.is_dir():
        return []
    return [build_dataset_catalog_item(path) for path in sorted(cfg.glob('*.txt'))]


def find_dataset_manifest(dataset_id: str) -> Path:
    target = (dataset_config_dir() / f'{dataset_id}.txt').resolve()
    cfg_root = dataset_config_dir().resolve()
    if cfg_root not in target.parents:
        raise HTTPException(status_code=400, detail='Invalid dataset id')
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail='Dataset not found')
    return target


def dataset_preview_rows(path: Path, limit: int = 25) -> tuple[list[str], int]:
    rows: list[str] = []
    total = 0
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            value = line.strip()
            if not value:
                continue
            total += 1
            if len(rows) < max(1, min(limit, 100)):
                rows.append(value)
    return rows, total


def dataset_detail(dataset_id: str) -> DatasetDetailOut:
    path = find_dataset_manifest(dataset_id)
    item = build_dataset_catalog_item(path)
    return DatasetDetailOut(**item.model_dump(), manifest_path=str(path))


def dataset_preview(dataset_id: str, limit: int = 25) -> DatasetPreviewOut:
    path = find_dataset_manifest(dataset_id)
    preview, total_rows = dataset_preview_rows(path, limit=limit)
    return DatasetPreviewOut(dataset_id=dataset_id, source=dataset_id.split('_', 1)[0], preview=preview, total_rows=total_rows)


def create_dataset_catalog(request: DatasetCreateRequest) -> DatasetDetailOut:
    cfg_dir = dataset_config_dir()
    cfg_dir.mkdir(parents=True, exist_ok=True)
    target = (cfg_dir / f'{request.dataset_id}.txt').resolve()
    cfg_root = cfg_dir.resolve()
    if cfg_root not in target.parents:
        raise HTTPException(status_code=400, detail='Invalid dataset id')
    if target.exists():
        raise HTTPException(status_code=409, detail=f'Dataset already exists: {request.dataset_id}')
    target.write_text('\n'.join(request.accessions) + '\n', encoding='utf-8')
    item = build_dataset_catalog_item(target)
    return DatasetDetailOut(**item.model_dump(), manifest_path=str(target))
