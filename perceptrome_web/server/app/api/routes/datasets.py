from fastapi import APIRouter, Depends

from ...deps import get_current_user_strict
from ...models import User
from ...schemas import DatasetCatalogItemOut, DatasetCreateRequest, DatasetDetailOut, DatasetPreviewOut
from ...services import dataset_service

router = APIRouter(prefix='/api/datasets', tags=['datasets'])


@router.get('', response_model=list[DatasetCatalogItemOut])
def list_datasets(user: User = Depends(get_current_user_strict)):
    _ = user
    return dataset_service.load_dataset_catalog()


@router.post('', response_model=DatasetDetailOut, status_code=201)
def create_dataset(payload: DatasetCreateRequest, user: User = Depends(get_current_user_strict)):
    _ = user
    return dataset_service.create_dataset_catalog(payload)


@router.get('/{dataset_id}', response_model=DatasetDetailOut)
def get_dataset(dataset_id: str, user: User = Depends(get_current_user_strict)):
    _ = user
    return dataset_service.dataset_detail(dataset_id)


@router.get('/{dataset_id}/preview', response_model=DatasetPreviewOut)
def preview_dataset(dataset_id: str, limit: int = 25, user: User = Depends(get_current_user_strict)):
    _ = user
    return dataset_service.dataset_preview(dataset_id, limit=limit)
