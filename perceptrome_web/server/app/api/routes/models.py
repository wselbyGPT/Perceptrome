from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ...deps import get_current_user_strict, get_db
from ...models import User
from ...schemas import (
    ModelRegisterFromRunRequest,
    ModelRegistrySummaryOut,
    ModelUpdateRequest,
    ModelVersionUpdateRequest,
    RegisteredModelOut,
)
from ...services import model_registry_service

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=list[RegisteredModelOut])
def list_models(
    search: str | None = None,
    architecture: str | None = None,
    status: str | None = None,
    visibility: str | None = None,
    limit: int = 100,
    user: User = Depends(get_current_user_strict),
    db: Session = Depends(get_db),
):
    return model_registry_service.list_models(
        db,
        user,
        search=search,
        architecture=architecture,
        status=status,
        visibility=visibility,
        limit=limit,
    )


@router.get("/summary", response_model=ModelRegistrySummaryOut)
def model_summary(user: User = Depends(get_current_user_strict), db: Session = Depends(get_db)):
    return model_registry_service.registry_summary(db, user)


@router.post("/register-from-run", response_model=RegisteredModelOut)
def register_model_from_run(
    payload: ModelRegisterFromRunRequest,
    user: User = Depends(get_current_user_strict),
    db: Session = Depends(get_db),
):
    return model_registry_service.register_from_run(
        db,
        user,
        run_id=payload.run_id,
        model_id=payload.model_id,
        name=payload.name,
        description=payload.description,
        visibility=payload.visibility,
        tags=payload.tags,
        version_label=payload.version_label,
        version_status=payload.version_status,
    )


@router.get("/{model_id}", response_model=RegisteredModelOut)
def get_model(model_id: str, user: User = Depends(get_current_user_strict), db: Session = Depends(get_db)):
    model = model_registry_service.get_model(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    model_registry_service.assert_model_view(model, user)
    return model_registry_service.model_to_out(model)


@router.patch("/{model_id}", response_model=RegisteredModelOut)
def update_model(
    model_id: str,
    payload: ModelUpdateRequest,
    user: User = Depends(get_current_user_strict),
    db: Session = Depends(get_db),
):
    return model_registry_service.update_model(db, user, model_id, payload.model_dump(exclude_unset=True))


@router.post("/{model_id}/versions/from-run", response_model=RegisteredModelOut)
def register_model_version_from_run(
    model_id: str,
    payload: ModelRegisterFromRunRequest,
    user: User = Depends(get_current_user_strict),
    db: Session = Depends(get_db),
):
    return model_registry_service.register_from_run(
        db,
        user,
        run_id=payload.run_id,
        model_id=model_id,
        name=payload.name,
        description=payload.description,
        visibility=payload.visibility,
        tags=payload.tags,
        version_label=payload.version_label,
        version_status=payload.version_status,
    )


@router.patch("/{model_id}/versions/{version_id}", response_model=RegisteredModelOut)
def update_model_version(
    model_id: str,
    version_id: str,
    payload: ModelVersionUpdateRequest,
    user: User = Depends(get_current_user_strict),
    db: Session = Depends(get_db),
):
    return model_registry_service.update_version(db, user, model_id, version_id, payload.model_dump(exclude_unset=True))


@router.get("/{model_id}/versions/{version_id}/artifacts/{artifact_id}/download")
def download_model_artifact(
    model_id: str,
    version_id: str,
    artifact_id: int,
    user: User = Depends(get_current_user_strict),
    db: Session = Depends(get_db),
):
    return model_registry_service.download_model_artifact(db, user, model_id, version_id, artifact_id)
