from collections import deque
from pathlib import Path
from queue import Queue
from threading import Thread

import cv2
from src.errors.clip import ClipWriteError


class _ClipWriter:
    """Encodes the frames of one clip in a background thread."""

    def __init__(self, writer):
        self.writer = writer
        self.frames = Queue()
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while (frame := self.frames.get()) is not None:
            self.writer.write(frame)
        self.writer.release()

    def put(self, frame):
        self.frames.put(frame)

    def finish(self):
        self.frames.put(None)


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

        self.clip = None
        self.frames_remaining = 0
        self.finishing = []

    def _resize(self, frame):
        height, width = frame.shape[:2]
        if width == self.width:
            return frame
        new_height = int(height * self.width / width)
        return cv2.resize(frame, (self.width, new_height))

    def add_frame(self, frame):
        small = self._resize(frame)
        self.buffer.append(small)

        if self.clip is not None:
            self.clip.put(small)
            self.frames_remaining -= 1
            if self.frames_remaining == 0:
                self._stop_clip()

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

        writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, height))

        if not writer.isOpened():
            raise ClipWriteError(f"Could not open clip file for writing: {output_path}")

        self._stop_clip()
        self.clip = _ClipWriter(writer)
        for frame in self.buffer:
            self.clip.put(frame)
        self.frames_remaining = self.frames_after

    def _stop_clip(self):
        clip, self.clip = self.clip, None
        if clip is not None:
            clip.finish()
            self.finishing.append(clip)
        self.finishing = [c for c in self.finishing if c.thread.is_alive()]

    def close(self):
        self._stop_clip()
        for clip in self.finishing:
            clip.thread.join()
        self.finishing.clear()
