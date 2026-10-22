import csv
import math
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BIN_STEPS = (0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600)
MAX_BARS = 30
OK_COLOR = "tab:blue"
ALERT_COLOR = "red"
LIMIT_COLOR = "black"


def _read_series(series_path):
    timestamps = []
    counts = []
    with open(series_path, newline="") as file:
        for row in csv.DictReader(file):
            timestamps.append(float(row["timestamp"]))
            counts.append(int(row["count"]))
    return timestamps, counts


def choose_bin_seconds(duration, max_bars=MAX_BARS):
    for step in BIN_STEPS:
        if duration / step <= max_bars:
            return step
    return BIN_STEPS[-1]


def build_bars(timestamps, counts, bin_seconds):
    times = np.asarray(timestamps, dtype=float)
    values = np.asarray(counts, dtype=float)
    indices = np.floor(times / bin_seconds).astype(int)
    starts = []
    heights = []
    for index in np.unique(indices):
        starts.append(index * bin_seconds)
        heights.append(int(np.floor(np.median(values[indices == index]) + 0.5)))
    return np.array(starts), np.array(heights)


def set_integer_yaxis(axes, heights, limit, max_ticks=12):
    top = int(max(max(heights, default=0), limit)) + 1
    step = max(1, math.ceil(top / max_ticks))
    axes.set_yticks(range(0, top + 1, step))
    axes.set_ylim(0, top)


def _decorate(axes, limit):
    axes.axhline(limit, color=LIMIT_COLOR, linestyle="--", linewidth=1.5, zorder=3)
    axes.set_xlabel("Time (s)")
    axes.set_ylabel("People in zone")
    axes.set_title("Redzone occupancy")

    figure = axes.figure
    for legend in list(figure.legends):
        legend.remove()
    engine = figure.get_layout_engine()
    constrained = (
        engine is not None and type(engine).__name__ == "ConstrainedLayoutEngine"
    )
    figure.legend(
        handles=[
            Patch(color=OK_COLOR, label="Within limit"),
            Patch(color=ALERT_COLOR, label="Above limit"),
            Line2D(
                [0], [0], color=LIMIT_COLOR, linestyle="--", label=f"Limit ({limit})"
            ),
        ],
        loc="outside center right" if constrained else "center right",
        ncol=1,
        frameon=False,
    )


def draw_empty_occupancy(axes, limit):
    axes.set_xlim(0, 1)
    set_integer_yaxis(axes, [], limit)
    _decorate(axes, limit)


def draw_occupancy(axes, timestamps, counts, limit, bin_seconds=None):
    bin_seconds = bin_seconds or choose_bin_seconds(max(timestamps))
    starts, heights = build_bars(timestamps, counts, bin_seconds)
    colors = [ALERT_COLOR if height > limit else OK_COLOR for height in heights]
    bars = axes.bar(
        starts + bin_seconds / 2, heights, width=bin_seconds * 0.85, color=colors
    )
    axes.bar_label(bars, labels=[str(height) for height in heights], padding=2)
    axes.set_xlim(0, max(starts) + bin_seconds)
    set_integer_yaxis(axes, heights, limit)
    _decorate(axes, limit)


def plot_occupancy(series_path, output_path, limit, bin_seconds=None):
    timestamps, counts = _read_series(series_path)
    bin_seconds = bin_seconds or choose_bin_seconds(max(timestamps))
    bars = math.ceil(max(timestamps) / bin_seconds) + 1

    figure = Figure(figsize=(max(8, min(16, 0.4 * bars + 3)), 5), layout="constrained")
    axes = figure.subplots()
    draw_occupancy(axes, timestamps, counts, limit, bin_seconds)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
