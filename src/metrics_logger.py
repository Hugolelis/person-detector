import csv
from pathlib import Path


class MetricsLogger:
    def __init__(self, output_dir, fps: float):
        self.fps = fps
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._series_file = open(output_dir / "series.csv", "w", newline="")
        self._series_writer = csv.writer(self._series_file)
        self._series_writer.writerow(["frame", "timestamp", "count", "in_violation"])

        self._events_file = open(output_dir / "events.csv", "w", newline="")
        self._events_writer = csv.writer(self._events_file)
        self._events_writer.writerow(
            [
                "start_frame",
                "start_time",
                "end_frame",
                "end_time",
                "peak_count",
                "clip_path",
            ]
        )

        self._open_violation = None
        self._last_frame = 0

    def _to_seconds(self, frame_index: int) -> float:
        return round(frame_index / self.fps, 3)

    def log_frame(self, frame_index: int, count: int, in_violation: bool):
        self._last_frame = frame_index
        self._series_writer.writerow(
            [frame_index, self._to_seconds(frame_index), count, in_violation]
        )
        if self._open_violation is not None:
            self._open_violation["peak"] = max(self._open_violation["peak"], count)

    def start_violation(self, frame_index: int, clip_path):
        self._open_violation = {
            "start_frame": frame_index,
            "clip_path": str(clip_path),
            "peak": 0,
        }

    def end_violation(self, frame_index: int):
        violation = self._open_violation
        if violation is None:
            return
        self._events_writer.writerow(
            [
                violation["start_frame"],
                self._to_seconds(violation["start_frame"]),
                frame_index,
                self._to_seconds(frame_index),
                violation["peak"],
                violation["clip_path"],
            ]
        )
        self._open_violation = None

    def close(self):
        if self._open_violation is not None:
            self.end_violation(self._last_frame)
        self._series_file.close()
        self._events_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()