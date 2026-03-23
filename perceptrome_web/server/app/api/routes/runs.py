from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...deps import get_current_user_strict, get_db
from ...models import Run, RunArtifact, User
from ...schemas import BioASTVisualizationBundleOut, RunArtifactOut, RunLineageOut, RunOut, RunsBoardOut, RunStartRequest, RunSummaryOut
from ...services import run_service

router = APIRouter(prefix='/api/runs', tags=['runs'])


@router.post('/start')
def start_run(payload: RunStartRequest | None = None, user: User = Depends(get_current_user_strict), db: Session = Depends(get_db)):
    cfg = dict((payload.config if payload else {}) or {})
    return run_service.execute_run(cfg=cfg, user=user, db=db)


@router.post('/{run_id}/replay')
def replay_run(run_id: str, user: User = Depends(get_current_user_strict), db: Session = Depends(get_db)):
    source_run = run_service.find_run(db, run_id)
    if not source_run:
        raise HTTPException(status_code=404, detail='Run not found')
    run_service.assert_run_access(source_run, user)
    descriptor, descriptor_hash = run_service.load_run_replay_descriptor(source_run)
    required_inputs = descriptor.get('required_input_artifacts')
    if not isinstance(required_inputs, list):
        raise HTTPException(status_code=400, detail='Replay descriptor is invalid: required_input_artifacts must be a list')
    if any(not isinstance(item, dict) for item in required_inputs):
        raise HTTPException(status_code=400, detail='Replay descriptor is invalid: each required_input_artifacts entry must be an object')
    run_service.validate_required_inputs(required_inputs)
    explicit_params = descriptor.get('explicit_params')
    if not isinstance(explicit_params, dict):
        raise HTTPException(status_code=400, detail='Replay descriptor is invalid: explicit_params must be an object')
    replay_run_id = f'replay_{uuid4().hex}'
    replay_cfg = {'run_id': replay_run_id, 'manifest_id': replay_run_id, 'kind': descriptor.get('run_kind'), 'config_path': descriptor.get('config_path'), 'params': explicit_params}
    metadata = {'replayed_from_run_id': run_id, 'descriptor_hash': descriptor_hash}
    return run_service.execute_run(cfg=replay_cfg, user=user, db=db, manifest_metadata=metadata)


@router.get('', response_model=list[RunOut])
def list_runs(user: User = Depends(get_current_user_strict), db: Session = Depends(get_db), limit: int = 50):
    runs = db.execute(run_service.scoped_runs_query(user).order_by(Run.submitted_at.desc()).limit(max(1, min(limit, 200)))).scalars().all()
    return [run_service.run_to_out(run) for run in runs]


@router.get('/summary', response_model=RunSummaryOut)
def runs_summary(user: User = Depends(get_current_user_strict), db: Session = Depends(get_db)):
    runs = db.execute(run_service.scoped_runs_query(user).order_by(Run.submitted_at.desc()).limit(1000)).scalars().all()
    return run_service.runs_summary_payload(runs)


@router.get('/active', response_model=RunsBoardOut)
def active_runs(user: User = Depends(get_current_user_strict), db: Session = Depends(get_db), limit: int = 12):
    runs = db.execute(run_service.scoped_runs_query(user).where(Run.state.in_(['queued', 'running'])).order_by(Run.submitted_at.desc()).limit(max(1, min(limit, 100)))).scalars().all()
    return run_service.runs_board(runs)


@router.get('/failures', response_model=RunsBoardOut)
def failed_runs(user: User = Depends(get_current_user_strict), db: Session = Depends(get_db), limit: int = 12):
    runs = db.execute(run_service.scoped_runs_query(user).where(Run.state == 'failed').order_by(Run.finished_at.desc(), Run.submitted_at.desc()).limit(max(1, min(limit, 100)))).scalars().all()
    return run_service.runs_board(runs)


@router.get('/{run_id}', response_model=RunOut)
def get_run(run_id: str, user: User = Depends(get_current_user_strict), db: Session = Depends(get_db)):
    run = run_service.find_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail='Run not found')
    run_service.assert_run_access(run, user)
    return run_service.run_to_out(run)


@router.get('/{run_id}/lineage', response_model=RunLineageOut)
def get_run_lineage(run_id: str, depth: int = 2, artifact_type: str | None = None, run_state: str | None = None, user: User = Depends(get_current_user_strict), db: Session = Depends(get_db)):
    run = run_service.find_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail='Run not found')
    run_service.assert_run_access(run, user)
    accessible_runs = db.execute(run_service.scoped_runs_query(user).order_by(Run.submitted_at.desc()).limit(400)).scalars().all()
    depth_limit = max(0, min(depth, 6))
    nodes, edges = run_service.build_lineage_graph(run=run, depth_limit=depth_limit, accessible_runs=accessible_runs)
    states = {item.strip().lower() for item in (run_state or '').split(',') if item.strip()}
    nodes, edges = run_service.filter_lineage_graph(nodes=nodes, edges=edges, root_run_id=run_id, artifact_type_filter=artifact_type, run_state_filter=states)
    return RunLineageOut(run_id=run_id, depth_limit=depth_limit, artifact_type_filter=artifact_type, run_state_filter=sorted(states), nodes=nodes, edges=edges)


@router.get('/{run_id}/bio-ast', response_model=BioASTVisualizationBundleOut)
def get_run_bio_ast_bundle(run_id: str, artifact_id: int | None = None, accession: str | None = None, user: User = Depends(get_current_user_strict), db: Session = Depends(get_db)):
    run = run_service.find_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail='Run not found')
    run_service.assert_run_access(run, user)
    return run_service.resolve_bio_ast_bundle(run, artifact_id=artifact_id, accession=accession)


@router.get('/{run_id}/artifacts', response_model=list[RunArtifactOut])
def list_run_artifacts(run_id: str, user: User = Depends(get_current_user_strict), db: Session = Depends(get_db)):
    run = run_service.find_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail='Run not found')
    run_service.assert_run_access(run, user)
    return run_service.run_to_out(run).artifacts


@router.get('/{run_id}/artifacts/{artifact_id}/download')
def download_artifact(run_id: str, artifact_id: int, user: User = Depends(get_current_user_strict), db: Session = Depends(get_db)):
    run = run_service.find_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail='Run not found')
    run_service.assert_run_access(run, user)
    artifact = db.execute(select(RunArtifact).where(RunArtifact.id == artifact_id).where(RunArtifact.run_id == run.id)).scalar_one_or_none()
    if not artifact:
        raise HTTPException(status_code=404, detail='Artifact not found')
    return run_service.download_artifact_response(run, artifact)


@router.get('/{run_id}/artifacts/download-by-path')
def download_artifact_by_path(run_id: str, path: str, user: User = Depends(get_current_user_strict), db: Session = Depends(get_db)):
    run = run_service.find_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail='Run not found')
    run_service.assert_run_access(run, user)
    return run_service.download_path_response(run, path)
