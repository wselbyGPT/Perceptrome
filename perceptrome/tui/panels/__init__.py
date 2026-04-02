"""Panel widgets for Perceptrome's TUI views."""

from .artifacts import ArtifactsPanel
from .base import BasePanel
from .config import ConfigPanel
from .data import DataPanel
from .events import EventsPanel
from .generate import GeneratePanel
from .history import HistoryPanel
from .jobs import JobsPanel
from .metrics import MetricsPanel
from .overview import OverviewPanel
from .pipeline import PipelinePanel
from .train import TrainPanel
from .troubleshoot import TroubleshootPanel

ALL_PANELS = [
    OverviewPanel,
    ConfigPanel,
    DataPanel,
    TrainPanel,
    GeneratePanel,
    HistoryPanel,
    ArtifactsPanel,
    JobsPanel,
    MetricsPanel,
    PipelinePanel,
    TroubleshootPanel,
    EventsPanel,
]

__all__ = [
    "ALL_PANELS",
    "BasePanel",
    "OverviewPanel",
    "ConfigPanel",
    "DataPanel",
    "TrainPanel",
    "GeneratePanel",
    "HistoryPanel",
    "ArtifactsPanel",
    "JobsPanel",
    "MetricsPanel",
    "PipelinePanel",
    "TroubleshootPanel",
    "EventsPanel",
]
