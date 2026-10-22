import cv2
import numpy as np
import pytest
from src.errors.video import VideoOpenError
from src.reporting.run_dir import make_run_dir
from src.video.clip_recorder import ClipRecorder
from src.video.reader import DEFAULT_LIVE_FPS, VideoReader
from tests.test_pipeline import make_frame


def test_a_camera_is_opened_by_index_and_flagged_as_live(fake_camera):
    with VideoReader(0) as reader:
        assert reader.is_live
        assert reader.path is None
        assert (reader.width, reader.height) == (320, 240)
        frames = reader.frames()
        assert next(frames).shape == (240, 320, 3)
        assert next(frames).shape == (240, 320, 3)


def test_a_camera_without_fps_falls_back_to_the_default(fake_camera):
    with VideoReader(0) as reader:
        assert reader.fps == DEFAULT_LIVE_FPS


def test_a_missing_camera_raises_a_clear_error(fake_camera):
    fake_camera.broken_indexes = {3}
    with pytest.raises(VideoOpenError, match="camera 3"):
        VideoReader(3)


def test_files_are_not_live(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    with VideoReader(path) as reader:
        assert not reader.is_live


def test_webcam_runs_get_their_own_folder_name():
    assert make_run_dir("outputs", 0).name.startswith("webcam_")
    assert make_run_dir("outputs", "videos/street.mp4").name.startswith("street_")


def clip_properties(path):
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = 0
    while capture.read()[0]:
        frames += 1
    capture.release()
    return fps, frames


def test_clip_length_follows_time_when_frames_arrive_slowly(tmp_path):
    recorder = ClipRecorder(fps=30, seconds_before=1, seconds_after=1, width=160)
    for index in range(10):
        recorder.add_frame(make_frame(), time_s=index * 0.2)
    path = tmp_path / "clip.mp4"
    recorder.start(path)
    for index in range(10, 20):
        recorder.add_frame(make_frame(), time_s=index * 0.2)
    recorder.close()
    fps, frames = clip_properties(path)
    assert fps == pytest.approx(5, abs=0.5)
    assert frames == 5 + 5
