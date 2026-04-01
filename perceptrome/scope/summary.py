from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class ValueRange:
    min_value: float
    max_value: float


@dataclass(frozen=True)
class RollingDistribution:
    bins: tuple[float, ...]
    counts: tuple[int, ...]
    rolling_counts: tuple[int, ...]


@dataclass(frozen=True)
class MetricBands:
    low: int
    mid: int
    high: int
    thresholds: tuple[float, float]


@dataclass(frozen=True)
class ProgressCounters:
    steps_done: int
    steps_target: int
    global_step: int

    @property
    def ratio(self) -> float:
        if self.steps_target <= 0:
            return 0.0
        return min(1.0, max(0.0, self.steps_done / float(self.steps_target)))


@dataclass(frozen=True)
class ScopeSummaryFrame:
    accession: str
    num_windows: int
    window_size: int
    stride: int
    start_idx: int
    end_idx: int
    status: str
    error_range: ValueRange
    metric_range: ValueRange
    error_distribution: RollingDistribution
    metric_distribution: RollingDistribution
    error_bands: MetricBands
    metric_bands: MetricBands
    progress: ProgressCounters


def _value_range(values: np.ndarray) -> ValueRange:
    if values.size == 0:
        return ValueRange(0.0, 1.0)
    return ValueRange(float(np.min(values)), float(np.max(values)))


def _rolling_distribution(values: np.ndarray, bins: int = 12, rolling_window: int = 256) -> RollingDistribution:
    if values.size == 0:
        bin_edges = np.linspace(0.0, 1.0, bins + 1, dtype=np.float64)
        empty = tuple(0 for _ in range(bins))
        return RollingDistribution(tuple(float(v) for v in bin_edges), empty, empty)

    min_v = float(np.min(values))
    max_v = float(np.max(values))
    if max_v <= min_v:
        max_v = min_v + 1e-8
    bin_edges = np.linspace(min_v, max_v, bins + 1, dtype=np.float64)
    total_counts, _ = np.histogram(values, bins=bin_edges)
    rolling_slice = values[-min(values.size, rolling_window) :]
    rolling_counts, _ = np.histogram(rolling_slice, bins=bin_edges)
    return RollingDistribution(
        tuple(float(v) for v in bin_edges),
        tuple(int(v) for v in total_counts.tolist()),
        tuple(int(v) for v in rolling_counts.tolist()),
    )


def _metric_bands(values: np.ndarray) -> MetricBands:
    if values.size == 0:
        return MetricBands(low=0, mid=0, high=0, thresholds=(0.0, 1.0))
    q1, q2 = np.quantile(values, [0.33, 0.66])
    low = int(np.sum(values <= q1))
    mid = int(np.sum((values > q1) & (values <= q2)))
    high = int(np.sum(values > q2))
    return MetricBands(low=low, mid=mid, high=high, thresholds=(float(q1), float(q2)))


def build_scope_summary_frame(
    *,
    accession: str,
    errors: np.ndarray,
    metric_values: np.ndarray,
    window_size: int,
    stride: int,
    start_idx: int,
    width: int,
    status: str,
    steps_done: int = 0,
    steps_target: int = 0,
    global_step: int = 0,
    bins: int = 12,
    rolling_window: int = 256,
) -> ScopeSummaryFrame:
    num_windows = int(errors.shape[0])
    end_idx = min(start_idx + width, num_windows)
    return ScopeSummaryFrame(
        accession=accession,
        num_windows=num_windows,
        window_size=window_size,
        stride=stride,
        start_idx=start_idx,
        end_idx=end_idx,
        status=status,
        error_range=_value_range(errors),
        metric_range=_value_range(metric_values),
        error_distribution=_rolling_distribution(errors, bins=bins, rolling_window=rolling_window),
        metric_distribution=_rolling_distribution(metric_values, bins=bins, rolling_window=rolling_window),
        error_bands=_metric_bands(errors),
        metric_bands=_metric_bands(metric_values),
        progress=ProgressCounters(
            steps_done=int(steps_done),
            steps_target=int(steps_target),
            global_step=int(global_step),
        ),
    )


def normalize_values(values: np.ndarray, value_range: ValueRange) -> np.ndarray:
    if values.size == 0:
        return values
    span = max(value_range.max_value - value_range.min_value, 1e-8)
    return (values - value_range.min_value) / span


def build_gradient_row(values: np.ndarray, *, start_idx: int, end_idx: int, width: int, palette: str = " .:-=+*#%@") -> str:
    if width <= 1 or end_idx <= start_idx or values.size == 0:
        return ""
    chars: list[str] = []
    max_cols = max(0, width - 1)
    for col, wi in enumerate(range(start_idx, end_idx)):
        if col >= max_cols:
            break
        val = float(min(1.0, max(0.0, values[wi])))
        idx = int(val * (len(palette) - 1))
        chars.append(palette[idx])
    return "".join(chars)


class ScopeSummaryAdapter:
    """Poll/subscribe adapter around summary frame production."""

    def __init__(self) -> None:
        self._latest: ScopeSummaryFrame | None = None
        self._subscribers: list[Callable[[ScopeSummaryFrame], None]] = []

    def publish(self, frame: ScopeSummaryFrame) -> ScopeSummaryFrame:
        self._latest = frame
        for callback in tuple(self._subscribers):
            callback(frame)
        return frame

    def poll(self) -> ScopeSummaryFrame | None:
        return self._latest

    def subscribe(self, callback: Callable[[ScopeSummaryFrame], None]) -> Callable[[], None]:
        self._subscribers.append(callback)

        def _unsubscribe() -> None:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

        return _unsubscribe

    def stream(self, frames: Iterable[ScopeSummaryFrame]) -> None:
        for frame in frames:
            self.publish(frame)
