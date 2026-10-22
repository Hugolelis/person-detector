import cv2
import numpy as np
import pytest
from src.errors.video import VideoOpenError
from src.gui.source_preview import describe_source, load_preview


def write_video(path, frames):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(frames):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    return path


def test_a_file_preview_has_its_first_frame_and_fps(tmp_path):
    preview = load_preview(write_video(tmp_path / "clip.mp4", 3))
    assert not preview.live
    assert preview.fps == 10
    assert preview.first_frame.shape == (48, 64, 3)


def test_a_missing_file_raises_a_video_error(tmp_path):
    with pytest.raises(VideoOpenError):
        load_preview(tmp_path / "missing.mp4")


def test_a_camera_preview_skips_the_warm_up_frames(fake_camera):
    preview = load_preview(0)
    assert preview.live
    assert preview.first_frame.max() == 5  # the fifth frame the fake camera sent


def test_sources_are_described_for_the_panel():
    assert describe_source("videos/street.mp4", live=False) == "street.mp4"
    assert describe_source(0, live=True) == "Webcam (camera 0)"
