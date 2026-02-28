from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import uvicorn

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_STATIC_DIR = ROOT_DIR / "client" / "dist"
REPO_ROOT = ROOT_DIR.parent


def ensure_static_dir(static_dir: Path) -> Path:
    if not static_dir.exists():
        raise SystemExit(
            f"[perceptrome_web] client build not found: {static_dir}\n"
            "You probably need to run:\n\n"
            "    cd client\n"
            "    npm install\n"
            "    npm run build\n"
        )

    index_file = static_dir / "index.html"
    if not index_file.is_file():
        raise SystemExit(
            f"[perceptrome_web] {static_dir} exists but index.html is missing.\n"
            "Make sure your Vite build output is in client/dist."
        )

    return static_dir


async def _send_status(ws: WebSocket, status: str, progress: float | None = None) -> None:
    await ws.send_json({"type": "status", "status": status, "progress": progress})


async def _send_log(ws: WebSocket, message: str) -> None:
    await ws.send_json({"type": "log", "message": message})


async def _run_command(ws: WebSocket, command: str, cwd: str | None = None) -> dict[str, Any]:
    if not command.strip():
        raise ValueError("command cannot be empty")

    resolved_cwd = Path(cwd).resolve() if cwd else REPO_ROOT
    env = os.environ.copy()
    venv_bin = resolved_cwd / ".venv" / "bin"
    env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}" if venv_bin.exists() else env.get("PATH", "")

    await _send_status(ws, "running", 0.0)
    await _send_log(ws, f"[web] cwd={resolved_cwd}")
    await _send_log(ws, f"[web] $ {command}")

    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=str(resolved_cwd),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    try:
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            await _send_log(ws, line.decode(errors="replace").rstrip("\n"))

        exit_code = await proc.wait()
    except asyncio.CancelledError:
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        raise
    status = "done" if exit_code == 0 else "error"
    await _send_status(ws, status, 1.0 if exit_code == 0 else None)

    result = {
        "command": command,
        "cwd": str(resolved_cwd),
        "exit_code": exit_code,
        "ok": exit_code == 0,
    }
    await ws.send_json({"type": "result", "payload": result})
    return result


def create_app(static_dir: Path) -> FastAPI:
    app = FastAPI(title="Perceptrome Web")

    @app.websocket("/ws")
    async def perceptrome_ws(ws: WebSocket) -> None:
        await ws.accept()

        active_task: asyncio.Task[dict[str, Any]] | None = None

        async def stop_active(notify: bool = True) -> None:
            nonlocal active_task
            if active_task and not active_task.done():
                active_task.cancel()
                try:
                    await active_task
                except asyncio.CancelledError:
                    if notify:
                        await _send_log(ws, "[web] Active run cancelled.")
                except Exception as exc:  # pylint: disable=broad-except
                    if notify:
                        await _send_log(ws, f"[web] Error during cancellation: {exc!r}")
                if notify:
                    await _send_status(ws, "pending", 0.0)
            active_task = None

        try:
            await _send_status(ws, "pending", 0.0)
            await _send_log(ws, "Connected to Perceptrome backend.")

            while True:
                raw = await ws.receive_text()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    await _send_log(ws, "Error: message must be valid JSON.")
                    await _send_status(ws, "error", None)
                    continue

                msg_type = data.get("type")

                if msg_type == "start_run":
                    if active_task and not active_task.done():
                        await _send_log(ws, "Error: a run is already active.")
                        await _send_status(ws, "error", None)
                        continue

                    command = str(data.get("command") or "").strip()
                    cwd = data.get("cwd")
                    if not command:
                        await _send_log(ws, "Error: 'command' is required for start_run.")
                        await _send_status(ws, "error", None)
                        continue

                    active_task = asyncio.create_task(_run_command(ws, command, cwd))

                    def _done_callback(task: asyncio.Task[dict[str, Any]]) -> None:
                        nonlocal active_task
                        active_task = None
                        if task.cancelled():
                            return
                        if task.exception() is not None:
                            # Best effort; websocket might be closed.
                            try:
                                asyncio.create_task(_send_log(ws, f"Run failed: {task.exception()!r}"))
                                asyncio.create_task(_send_status(ws, "error", None))
                            except RuntimeError:
                                return

                    active_task.add_done_callback(_done_callback)

                elif msg_type == "stop_run":
                    await stop_active()

                else:
                    await _send_log(ws, f"Error: unsupported message type {msg_type!r}")
                    await _send_status(ws, "error", None)

        except WebSocketDisconnect:
            await stop_active(notify=False)

    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the Perceptrome web UI and WebSocket API.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="TCP port to bind")
    parser.add_argument(
        "--static-dir",
        type=str,
        default=str(DEFAULT_STATIC_DIR),
        help="Path to the built client/dist directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    static_dir = ensure_static_dir(Path(args.static_dir))
    app = create_app(static_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


try:
    _default_static_dir = ensure_static_dir(DEFAULT_STATIC_DIR)
    app = create_app(_default_static_dir)
except SystemExit:
    app = FastAPI(title="Perceptrome Web (no static build found)")


if __name__ == "__main__":
    main()
