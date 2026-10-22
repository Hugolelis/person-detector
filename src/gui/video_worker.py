from dataclasses import dataclass

import cv2
from PySide6.QtCore import QThread, Signal
from src.errors.base import PersonDetectorError
from src.pipeline import Pipeline
from src.video_reader import VideoReader

DISPLAY_WIDTH = 1280


@dataclass
class FrameUpdate:
    frame: object
    count: int
    time_s: float
    in_violation: bool


def shrink_for_display(frame):
    height, width = frame.shape[:2]
    if width <= DISPLAY_WIDTH:
        return frame
    new_height = round(height * DISPLAY_WIDTH / width)
    return cv2.resize(frame, (DISPLAY_WIDTH, new_height), interpolation=cv2.INTER_AREA)


class VideoWorker(QThread):
    frame_ready = Signal(object)
    error = Signal(str)

    def __init__(self, video_path, config, run_dir):
        super().__init__()
        self.video_path = video_path
        self.config = config
        self.run_dir = run_dir
        self._running = True
        self._paused = False

    def stop(self):
        self._running = False

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    @property
    def is_paused(self):
        return self._paused

    def run(self):
        processed = 0
        try:
            with (
                VideoReader(self.video_path) as reader,
                Pipeline(self.config, reader.fps, self.run_dir) as pipeline,
            ):
                for frame_index, frame in enumerate(reader.frames()):
                    while self._paused and self._running:
                        self.msleep(50)
                    if not self._running:
                        break
                    result = pipeline.process(frame, frame_index)
                    processed += 1
                    update = FrameUpdate(
                        frame=shrink_for_display(result.annotated_frame),
                        count=result.count,
                        time_s=round(frame_index / reader.fps, 3),
                        in_violation=result.in_violation,
                    )
                    self.frame_ready.emit(update)
            if processed:
                pipeline.plot_chart()
        except PersonDetectorError as e:
            self.error.emit(str(e))
