"""Panel widgets for Perceptrome's seven TUI views."""

from .base import BasePanel
from .diagnostics import DiagnosticsPanel
from .history import HistoryPanel
from .jobs import JobsPanel
from .metrics import MetricsPanel
from .overview import OverviewPanel
from .pipeline import PipelinePanel
from .settings import SettingsPanel

ALL_PANELS = [
    OverviewPanel,
    JobsPanel,
    MetricsPanel,
    PipelinePanel,
    HistoryPanel,
    DiagnosticsPanel,
    SettingsPanel,
]

__all__ = [
    "ALL_PANELS",
    "BasePanel",
    "OverviewPanel",
    "JobsPanel",
    "MetricsPanel",
    "PipelinePanel",
    "HistoryPanel",
    "DiagnosticsPanel",
    "SettingsPanel",
]
