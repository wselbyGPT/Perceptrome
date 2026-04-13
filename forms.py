from __future__ import annotations

from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Checkbox, Input, Select, Static


class LabeledInput(Vertical):
    def __init__(self, label: str, value: str = "", *, input_id: str) -> None:
        super().__init__()
        self._label = label
        self._value = value
        self._input_id = input_id

    def compose(self):
        yield Static(self._label)
        yield Input(value=self._value, id=self._input_id)


class ValidationBanner(Static):
    def show_error(self, message: str) -> None:
        self.update(f"[error] {message}")

    def clear(self) -> None:
        self.update("")


def parse_int(value: str, *, field: str) -> int:
    try:
        return int(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{field} must be an integer") from exc


def parse_float(value: str, *, field: str) -> float:
    try:
        return float(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{field} must be a number") from exc


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def choice(id: str, options: list[tuple[str, str]], *, value: str | None = None) -> Select:
    return Select(options=options, value=value or (options[0][1] if options else None), id=id)


def submit_reset_row(*, submit_id: str = "form-submit", reset_id: str = "form-reset") -> Horizontal:
    row = Horizontal()
    row.mount(Button("Submit", id=submit_id, variant="success"))
    row.mount(Button("Reset", id=reset_id))
    return row


def bool_toggle(label: str, *, toggle_id: str, value: bool = False) -> Checkbox:
    return Checkbox(label, value=value, id=toggle_id)
