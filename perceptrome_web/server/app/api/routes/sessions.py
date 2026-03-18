from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ...core.db import SessionLocal
from ...services import run_service, session_service

router = APIRouter(tags=['sessions'])


@router.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    db = SessionLocal()
    try:
        user = session_service.websocket_auth_user(websocket, db)
        if not user:
            await websocket.close(code=4401)
            return
        if user.must_change_password:
            await websocket.close(code=4403)
            return
        await run_service.websocket_run_loop(websocket, user, db)
    except WebSocketDisconnect:
        pass
    finally:
        db.close()
