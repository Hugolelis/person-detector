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


TIME_TOLERANCE = 1e-6


class ClipRecorder:
    def __init__(
        self,
        fps: float,
        seconds_before: float,
        seconds_after: float,
        width: int = 1280,
    ):
        """Keep the last seconds_before seconds and record seconds_after more.

        fps is only used to time frames added without an explicit time_s.
        """
        self.fps = fps
        self.width = width
        self.seconds_before = seconds_before
        self.seconds_after = seconds_after
        self.buffer = deque()
        self.frames_added = 0

        self.clip = None
        self.clip_end_time = 0.0
        self.finishing = []

    def _resize(self, frame):
        height, width = frame.shape[:2]
        if width == self.width:
            return frame
        new_height = int(height * self.width / width)
        return cv2.resize(frame, (self.width, new_height))

    def add_frame(self, frame, time_s=None):
        if time_s is None:
            time_s = self.frames_added / self.fps
        self.frames_added += 1
        small = self._resize(frame)

        self.buffer.append((time_s, small))
        oldest_allowed = time_s - self.seconds_before + TIME_TOLERANCE
        while len(self.buffer) > 1 and self.buffer[0][0] <= oldest_allowed:
            self.buffer.popleft()

        if self.clip is not None:
            self.clip.put(small)
            if time_s >= self.clip_end_time - TIME_TOLERANCE:
                self._stop_clip()

    def _recorded_fps(self):
        """Frame rate of the buffered frames, which is what the clip should play at."""
        first_time, last_time = self.buffer[0][0], self.buffer[-1][0]
        if len(self.buffer) < 2 or last_time <= first_time:
            return self.fps
        return (len(self.buffer) - 1) / (last_time - first_time)

    def start(self, output_path):
        output_path = Path(output_path)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ClipWriteError(
                f"Could not create output folder: {output_path.parent}"
            ) from e

        height, width = self.buffer[0][1].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            str(output_path), fourcc, self._recorded_fps(), (width, height)
        )

        if not writer.isOpened():
            raise ClipWriteError(f"Could not open clip file for writing: {output_path}")

        self._stop_clip()
        self.clip = _ClipWriter(writer)
        for _, frame in self.buffer:
            self.clip.put(frame)
        self.clip_end_time = self.buffer[-1][0] + self.seconds_after

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
