import os
import time

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class FakeCamera:
    """Stands in for cv2.VideoCapture(index): endless frames, no reported fps."""

    broken_indexes = set()
    delay = 0.02

    def __init__(self, index):
        self.index = index
        self.open = index not in FakeCamera.broken_indexes
        self.reads = 0

    def isOpened(self):
        return self.open

    def set(self, prop, value):
        return True

    def get(self, prop):
        sizes = {cv2.CAP_PROP_FRAME_WIDTH: 320, cv2.CAP_PROP_FRAME_HEIGHT: 240}
        return sizes.get(prop, 0.0)

    def read(self):
        time.sleep(FakeCamera.delay)
        self.reads += 1
        return True, np.full((240, 320, 3), self.reads % 255, dtype=np.uint8)

    def release(self):
        self.open = False


@pytest.fixture
def fake_camera(monkeypatch):
    """Camera indexes (ints) get FakeCamera, file paths still use the real capture."""
    FakeCamera.broken_indexes = set()
    FakeCamera.delay = 0.02
    real_capture = cv2.VideoCapture

    def capture(source):
        if isinstance(source, int):
            return FakeCamera(source)
        return real_capture(source)

    monkeypatch.setattr(cv2, "VideoCapture", capture)
    return FakeCamera
