import threading

import cv2
import numpy as np
from src.video.clip_recorder import ClipRecorder


def test_frames_already_at_the_target_width_are_not_resized():
    recorder = ClipRecorder(fps=10, seconds_before=1, seconds_after=1, width=160)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    assert recorder._resize(frame) is frame


def test_wider_frames_are_resized_keeping_the_proportion():
    recorder = ClipRecorder(fps=10, seconds_before=1, seconds_after=1, width=160)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    assert recorder._resize(frame).shape == (120, 160, 3)


def make_frame(value=0):
    return np.full((120, 160, 3), value, dtype=np.uint8)


def read_clip(path):
    capture = cv2.VideoCapture(str(path))
    count = 0
    while capture.read()[0]:
        count += 1
    capture.release()
    return count


def test_clip_has_buffered_and_following_frames_after_close(tmp_path):
    recorder = ClipRecorder(fps=10, seconds_before=1, seconds_after=1, width=160)
    for index in range(15):
        recorder.add_frame(make_frame(index))
    path = tmp_path / "clips" / "clip.mp4"
    recorder.start(path)
    for index in range(10):
        recorder.add_frame(make_frame(index))
    recorder.close()
    assert read_clip(path) == 20


def test_start_does_not_wait_for_the_buffer_to_be_encoded(tmp_path, monkeypatch):
    gate = threading.Event()
    written = []

    class SlowWriter:
        def isOpened(self):
            return True

        def write(self, frame):
            gate.wait()
            written.append(frame)

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoWriter", lambda *args: SlowWriter())
    recorder = ClipRecorder(fps=10, seconds_before=1, seconds_after=1, width=160)
    for index in range(10):
        recorder.add_frame(make_frame(index))
    recorder.start(tmp_path / "clip.mp4")
    for index in range(10):
        recorder.add_frame(make_frame(index))
    assert written == []
    gate.set()
    recorder.close()
    assert len(written) == 20


def test_close_without_a_clip_does_nothing():
    recorder = ClipRecorder(fps=10, seconds_before=1, seconds_after=1, width=160)
    recorder.add_frame(make_frame())
    recorder.close()
