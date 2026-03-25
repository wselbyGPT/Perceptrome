"""Textual application entrypoint for Perceptrome."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Footer, Header, TabbedContent, TabPane

from .diagnostics import capture_diagnostics
from .job_manager import JobManager
from .launcher import DEFAULT_COMMANDS
from .state_store import StateStore
from .panels import ALL_PANELS


class PerceptromeTUIApp(App[None]):
    """Main Perceptrome text UI shell with seven required views."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #workspace {
        height: 1fr;
    }
    """

    BINDINGS = [
        ("ctrl+p", "show_launcher", "Launcher"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.state = StateStore()
        self.jobs = JobManager()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="workspace"):
            with TabbedContent(initial="overview"):
                for panel_cls in ALL_PANELS:
                    with TabPane(panel_cls.TITLE, id=panel_cls.PANEL_ID):
                        yield panel_cls()
        yield Footer()

    def on_mount(self) -> None:
        diagnostics = capture_diagnostics()
        self.state.set_value("python", diagnostics.python_version)
        self.state.set_value("platform", diagnostics.platform)

    def action_show_launcher(self) -> None:
        command_lines = [f"{entry.label} ({entry.view_id})" for entry in DEFAULT_COMMANDS]
        self.notify("Launcher\n" + "\n".join(command_lines), title="Command Palette")


def main() -> None:
    PerceptromeTUIApp().run()


if __name__ == "__main__":
    main()
