from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from perceptrome.config import extract_configs, load_full_config
from perceptrome.encoding.parse import parse_fasta_sequence
from perceptrome.ncbi_fetch import fetch_fasta
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPdfWriter, QPen, QPolygon

from perceptrome.io_utils import read_catalog, select_unique_accessions, write_catalog

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_STATIC_DIR = ROOT_DIR / "client" / "dist"
REPO_ROOT = ROOT_DIR.parent


def _read_fasta_sequence(path: str) -> str:
    seq = parse_fasta_sequence(path)
    if not seq:
        raise ValueError(f"No sequence found in FASTA: {path}")
    return seq


def _default_pdf_output_for_mode(mode: str) -> str:
    return "generated/linear_genome.pdf" if mode.lower() == "linear" else "generated/circular_genome.pdf"


def _default_title_for_mode(mode: str, source: str) -> str:
    view_name = "Linear genome" if mode.lower() == "linear" else "Circular genome"
    return f"{view_name} ({source})"


def _write_circular_pdf(seq: str, output_path: str, title: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = QPdfWriter(output_path)
    writer.setPageSize(QPdfWriter.PageSize.A4)
    painter = QPainter(writer)
    try:
        rect = painter.viewport()
        cx = rect.width() // 2
        cy = rect.height() // 2
        radius = min(rect.width(), rect.height()) // 3
        gc_ratio = ((seq.count("G") + seq.count("C")) / max(1, len(seq)))

        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(QColor("#2f3b45"), 10))
        painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)

        arc_color = QColor("#20b2aa")
        painter.setPen(QPen(arc_color, 10))
        span = int(360 * 16 * gc_ratio)
        painter.drawArc(cx - radius, cy - radius, radius * 2, radius * 2, 90 * 16, -span)

        painter.setPen(QPen(QColor("#111111"), 1))
        painter.setFont(QFont("Sans Serif", 14))
        painter.drawText(cx - radius, cy - radius - 35, radius * 2, 30, Qt.AlignCenter, title)
        painter.setFont(QFont("Sans Serif", 11))
        painter.drawText(cx - radius, cy + radius + 15, radius * 2, 24, Qt.AlignCenter, f"Length: {len(seq):,} bp")
        painter.drawText(cx - radius, cy + radius + 38, radius * 2, 24, Qt.AlignCenter, f"GC: {gc_ratio * 100:.2f}%")
    finally:
        painter.end()


def _write_linear_pdf(seq: str, output_path: str, title: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = QPdfWriter(output_path)
    writer.setPageSize(QPdfWriter.PageSize.A4)
    painter = QPainter(writer)
    try:
        rect = painter.viewport()
        left = int(rect.width() * 0.08)
        top = rect.height() // 2
        width = int(rect.width() * 0.84)
        gc_ratio = ((seq.count("G") + seq.count("C")) / max(1, len(seq)))

        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(QColor("#2f3b45"), 8))
        painter.drawLine(left, top, left + width, top)

        gc_width = int(width * gc_ratio)
        painter.setPen(QPen(QColor("#20b2aa"), 8))
        painter.drawLine(left, top, left + gc_width, top)

        arrow = [
            QPoint(left + width, top),
            QPoint(left + width - 18, top - 12),
            QPoint(left + width - 18, top + 12),
        ]
        painter.setBrush(QColor("#2f3b45"))
        painter.drawPolygon(QPolygon(arrow))

        painter.setPen(QPen(QColor("#111111"), 1))
        painter.setFont(QFont("Sans Serif", 14))
        painter.drawText(left, top - 70, width, 30, Qt.AlignCenter, title)
        painter.setFont(QFont("Sans Serif", 11))
        painter.drawText(left, top + 30, width, 24, Qt.AlignCenter, f"Length: {len(seq):,} bp")
        painter.drawText(left, top + 53, width, 24, Qt.AlignCenter, f"GC: {gc_ratio * 100:.2f}%")
    finally:
        painter.end()


async def _build_view_pdf(payload: dict[str, Any]) -> dict[str, Any]:
    accession = str(payload.get("accession") or "").strip()
    fasta_path = str(payload.get("fasta_path") or "").strip()
    render_mode = str(payload.get("render_mode") or "circular").strip().lower()
    output_path = str(payload.get("output_path") or "").strip()
    title = str(payload.get("title") or "").strip()

    if render_mode not in {"circular", "linear"}:
        render_mode = "circular"

    source = ""
    if accession:
        cfg = load_full_config("stream_config.yaml")
        ncbi_cfg, _, io_cfg = extract_configs(cfg)
        fasta_path = fetch_fasta(accession, io_cfg, ncbi_cfg, force=False)
        source = f"accession {accession}"
    elif fasta_path:
        source = f"fasta {fasta_path}"
    else:
        raise ValueError("Provide a genome accession or FASTA path.")

    resolved_fasta = (REPO_ROOT / fasta_path).resolve() if not os.path.isabs(fasta_path) else Path(fasta_path).resolve()
    if not resolved_fasta.exists():
        raise ValueError(f"FASTA path does not exist: {fasta_path}")

    seq = _read_fasta_sequence(str(resolved_fasta))
    final_output = output_path or _default_pdf_output_for_mode(render_mode)
    output_file = (REPO_ROOT / final_output).resolve() if not os.path.isabs(final_output) else Path(final_output).resolve()
    title = title or _default_title_for_mode(render_mode, source)

    if render_mode == "linear":
        await asyncio.to_thread(_write_linear_pdf, seq, str(output_file), title)
    else:
        await asyncio.to_thread(_write_circular_pdf, seq, str(output_file), title)

    return {
        "ok": True,
        "status": f"Saved {render_mode} PDF -> {output_file}",
        "output_path": str(output_file.relative_to(REPO_ROOT)) if output_file.is_relative_to(REPO_ROOT) else str(output_file),
        "file_url": f"/generated-file?path={output_file}",
    }


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


def _build_dataset_catalog(payload: dict[str, Any]) -> dict[str, Any]:
    raw_quotas = payload.get("category_quotas")
    if not isinstance(raw_quotas, list) or not raw_quotas:
        raise ValueError("payload.category_quotas must be a non-empty list")

    category_quotas: list[tuple[str, int]] = []
    category_candidates: dict[str, list[str]] = {}
    logs: list[str] = []

    for idx, item in enumerate(raw_quotas):
        if not isinstance(item, dict):
            raise ValueError(f"category_quotas[{idx}] must be an object")
        category = str(item.get("category") or "").strip()
        source = str(item.get("source") or "").strip()
        count = int(item.get("count") or 0)
        if not category:
            raise ValueError(f"category_quotas[{idx}].category is required")
        if not source:
            raise ValueError(f"category_quotas[{idx}].source is required")
        if count <= 0:
            raise ValueError(f"category_quotas[{idx}].count must be > 0")

        candidates = read_catalog(source)
        category_quotas.append((category, count))
        category_candidates[category] = candidates
        logs.append(f"{category}: quota={count}, candidates={len(candidates)}, source={source}")

    shuffle_within_category = bool(payload.get("shuffle_within_category", False))
    output_catalog = str(payload.get("output_catalog") or "").strip()
    if not output_catalog:
        raise ValueError("payload.output_catalog is required")

    selected = select_unique_accessions(
        category_quotas,
        category_candidates,
        shuffle_within_category=shuffle_within_category,
    )
    if not selected:
        raise ValueError("No accessions were selected from the provided quotas")

    header = [
        "Generated by perceptrome_web create_dataset",
        f"total_accessions: {len(selected)}",
        f"shuffle_within_category: {shuffle_within_category}",
    ]
    write_catalog(output_catalog, selected, header=header)

    logs.insert(0, f"Wrote catalog: {output_catalog}")
    logs.insert(1, f"Selected accessions: {len(selected)}")

    return {
        "ok": True,
        "output_catalog": output_catalog,
        "selected_count": len(selected),
        "logs": logs,
    }


def create_app(static_dir: Path) -> FastAPI:
    app = FastAPI(title="Perceptrome Web")

    @app.get("/generated-file")
    async def generated_file(path: str = Query(..., description="Absolute or repo-relative file path")) -> FileResponse:
        resolved = (REPO_ROOT / path).resolve() if not os.path.isabs(path) else Path(path).resolve()
        if not resolved.exists() or not resolved.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        if not str(resolved).lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        return FileResponse(path=resolved, media_type="application/pdf", filename=resolved.name)

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

                elif msg_type == "create_dataset":
                    payload = data.get("payload")
                    if not isinstance(payload, dict):
                        await ws.send_json(
                            {
                                "type": "create_dataset_result",
                                "payload": {"ok": False, "error": "payload must be an object"},
                            }
                        )
                        continue

                    try:
                        result = _build_dataset_catalog(payload)
                    except Exception as exc:  # pylint: disable=broad-except
                        await ws.send_json(
                            {
                                "type": "create_dataset_result",
                                "payload": {"ok": False, "error": str(exc), "logs": [f"Create dataset failed: {exc}"]},
                            }
                        )
                        continue

                    await ws.send_json({"type": "create_dataset_result", "payload": result})

                elif msg_type == "view_generate_pdf":
                    payload = data.get("payload")
                    if not isinstance(payload, dict):
                        await ws.send_json({"type": "view_result", "payload": {"ok": False, "error": "payload must be an object"}})
                        continue

                    await ws.send_json({"type": "view_status", "status": "running"})
                    await ws.send_json({"type": "view_log", "message": "[view] Resolving genome source..."})
                    try:
                        result = await _build_view_pdf(payload)
                    except Exception as exc:  # pylint: disable=broad-except
                        await ws.send_json({"type": "view_log", "message": f"[view] ERROR: {exc}"})
                        await ws.send_json({"type": "view_status", "status": "error"})
                        await ws.send_json({"type": "view_result", "payload": {"ok": False, "error": str(exc)}})
                        continue

                    await ws.send_json({"type": "view_log", "message": result.get("status", "PDF generation complete.")})
                    await ws.send_json({"type": "view_status", "status": "done"})
                    await ws.send_json({"type": "view_result", "payload": result})

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
