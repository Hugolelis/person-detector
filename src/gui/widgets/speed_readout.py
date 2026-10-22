from time import perf_counter

from PySide6.QtWidgets import QLabel
from src.gui.fps_meter import FpsMeter, describe_speed


class SpeedReadout(QLabel):
    """Shows the source fps and how fast the processing really runs."""

    def __init__(self):
        super().__init__("")
        self.setWordWrap(True)
        self.meter = FpsMeter()
        self.live = False
        self.source_fps = None

    def _show(self, processing=None):
        lines = []
        if self.live:
            lines.append("Source: webcam (live)")
        elif self.source_fps:
            lines.append(f"Source video: {self.source_fps:.2f} fps")
        if processing:
            lines.append(f"Processing: {processing}")
        self.setText("\n".join(lines))

    def set_source(self, live, fps):
        self.live = live
        self.source_fps = None if live else fps
        self._show()

    def begin(self):
        self.meter.reset()
        self._show("measuring...")

    def tick(self):
        self.meter.tick(perf_counter())
        fps = self.meter.current_fps()
        self._show(describe_speed(fps, self.source_fps) if fps else "measuring...")

    def pause(self):
        self.meter.pause(perf_counter())
        self._show("paused")

    def resume(self):
        self.meter.resume(perf_counter())
        self._show("measuring...")

    def finish(self):
        average = self.meter.average_fps()
        if average:
            self._show(f"finished, {describe_speed(average, self.source_fps)}")
        else:
            self._show("stopped before the first frame")
