from __future__ import annotations


def submit(app: object) -> str | None:
    if hasattr(app, "submit_draft_job"):
        return app.submit_draft_job()
    return None


def rerun(app: object) -> bool:
    return bool(getattr(app, "rerun_selected_job", lambda: False)())


def clone(app: object) -> bool:
    return bool(getattr(app, "clone_selected_job_to_draft", lambda: False)())


def cancel(app: object) -> bool:
    return bool(getattr(app, "cancel_selected_job", lambda: False)())


def open_artifact(app: object, path: str | None = None) -> None:
    if hasattr(app, "open_artifact"):
        app.open_artifact(path)


def open_traceback(app: object) -> None:
    if hasattr(app, "action_show_traceback"):
        app.action_show_traceback()


def open_manifest(app: object) -> None:
    if hasattr(app, "open_manifest_for_selected"):
        app.open_manifest_for_selected()
