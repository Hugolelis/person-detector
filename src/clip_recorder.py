from collections import deque
from pathlib import Path

import cv2
from src.errors.clip import ClipWriteError


class ClipRecorder:
    def __init__(
        self,
        fps: float,
        seconds_before: float,
        seconds_after: float,
        width: int = 1280,
    ):
        self.fps = fps
        self.width = width
        self.frames_after = int(seconds_after * fps)
        self.buffer = deque(maxlen=int(seconds_before * fps))

        self.writer = None
        self.frames_remaining = 0

    def _resize(self, frame):
        height, width = frame.shape[:2]
        new_height = int(height * self.width / width)
        return cv2.resize(frame, (self.width, new_height))

    def add_frame(self, frame):
        small = self._resize(frame)
        self.buffer.append(small)

        if self.writer is not None:
            self.writer.write(small)
            self.frames_remaining -= 1
            if self.frames_remaining == 0:
                self.writer.release()
                self.writer = None

    def start(self, output_path):
        output_path = Path(output_path)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ClipWriteError(
                f"Could not create output folder: {output_path.parent}"
            ) from e

        height, width = self.buffer[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self.writer = cv2.VideoWriter(
            str(output_path), fourcc, self.fps, (width, height)
        )

        if not self.writer.isOpened():
            raise ClipWriteError(f"Could not open clip file for writing: {output_path}")

        for frame in self.buffer:
            self.writer.write(frame)
        self.frames_remaining = self.frames_after
