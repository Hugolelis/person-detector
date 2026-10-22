from dataclasses import dataclass
from pathlib import Path

from PySide6.QtWidgets import QFileDialog
from src.errors.video import VideoOpenError
from src.video.reader import VideoReader

VIDEO_FILTER = "Videos (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv *.webm);;All files (*)"
DEFAULT_VIDEO_DIR = "videos"
WEBCAM_INDEX = 0
WEBCAM_WARMUP_FRAMES = 5


@dataclass
class Preview:
    live: bool
    fps: float
    first_frame: object


def _read_preview_frame(reader):
    """The first frame of a file, or a settled frame of a camera."""
    frames = reader.frames()
    frame = None
    for _ in range(WEBCAM_WARMUP_FRAMES if reader.is_live else 1):
        frame = next(frames, None)
        if frame is None:
            break
    return frame


def load_preview(source) -> Preview:
    """Open source just long enough to read one frame. Raises on failure."""
    with VideoReader(source) as reader:
        live, fps = reader.is_live, reader.fps
        first_frame = _read_preview_frame(reader)
    if first_frame is None:
        raise VideoOpenError(f"No frames could be read from: {source}")
    return Preview(live, fps, first_frame)


def describe_source(source, live) -> str:
    return f"Webcam (camera {source})" if live else Path(source).name


def ask_video_path(parent, current_source) -> str:
    """Open the file dialog, starting where the current video is. "" if cancelled."""
    start_dir = DEFAULT_VIDEO_DIR
    if isinstance(current_source, str):
        start_dir = str(Path(current_source).parent)
    path, _ = QFileDialog.getOpenFileName(
        parent, "Select a video", start_dir, VIDEO_FILTER
    )
    return path
