"""Panel widgets for Perceptrome's seven TUI views."""

from .base import BasePanel
from .config import ConfigPanel
from .data import DataPanel
from .generate import GeneratePanel
from .history import HistoryPanel
from .overview import OverviewPanel
from .train import TrainPanel
from .troubleshoot import TroubleshootPanel

ALL_PANELS = [
    OverviewPanel,
    ConfigPanel,
    DataPanel,
    TrainPanel,
    GeneratePanel,
    HistoryPanel,
    TroubleshootPanel,
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
    "TroubleshootPanel",
]
