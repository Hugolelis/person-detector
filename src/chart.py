import csv
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure


def _read_series(series_path):
    timestamps = []
    counts = []
    with open(series_path, newline="") as file:
        for row in csv.DictReader(file):
            timestamps.append(float(row["timestamp"]))
            counts.append(int(row["count"]))
    return timestamps, counts


def plot_occupancy(series_path, output_path, limit):
    timestamps, counts = _read_series(series_path)

    times = np.linspace(timestamps[0], timestamps[-1], len(timestamps) * 20)
    values = np.interp(times, timestamps, counts)
    above_limit = np.ma.masked_where(values <= limit, values)

    figure = Figure(figsize=(12, 5))
    axes = figure.subplots()
    axes.plot(times, values, color="tab:blue", label="Occupancy")
    axes.plot(times, above_limit, color="red", label="Above limit")
    axes.axhline(limit, color="red", linestyle="--", label=f"Limit ({limit})")
    axes.legend()
    axes.set_xlabel("Time (s)")
    axes.set_ylabel("People in zone")
    axes.set_title("Redzone occupancy")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")