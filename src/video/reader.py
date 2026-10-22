from pathlib import Path

import cv2
from src.errors.video import VideoOpenError

DEFAULT_LIVE_FPS = 30.0


class VideoReader:
    def __init__(self, source: str | Path | int):
        """Open a video file, or a camera when source is its integer index."""
        self.is_live = isinstance(source, int)
        if self.is_live:
            self.path = None
            self.cap = cv2.VideoCapture(source)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.cap.isOpened():
                raise VideoOpenError(f"Could not open camera {source}")
        else:
            self.path = Path(source)
            self.cap = cv2.VideoCapture(str(self.path))
            if not self.cap.isOpened():
                raise VideoOpenError(f"Could not open video file: {self.path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.is_live and not self.fps > 1:
            self.fps = DEFAULT_LIVE_FPS
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frames(self):
        """
        Iterate over the video frames, releasing the capture when done.

        Yields:
            numpy.ndarray: the next BGR frame read from the video.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()
