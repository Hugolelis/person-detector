from threading import Event, Lock, Thread

from src.errors.base import PersonDetectorError
from src.errors.depth import DepthModelError


class DepthRunner:
    """Estimates depth in a background thread so detection never waits for it."""

    def __init__(self, estimator, stride: int = 10, asynchronous: bool = True):
        """Estimate on every stride-th submitted frame, skipping while still busy."""
        self.estimator = estimator
        self.stride = max(1, stride)
        self.asynchronous = asynchronous
        self.submitted = 0
        self._latest = None
        self._error = None
        self._pending = None
        self._busy = False
        self._lock = Lock()
        self._wake = Event()
        self._stopped = False
        self._thread = None
        if asynchronous:
            self._thread = Thread(target=self._run, daemon=True)
            self._thread.start()

    def _estimate(self, frame):
        try:
            depth = self.estimator.estimate(frame)
        except PersonDetectorError as e:
            self._error = e
        except Exception as e:
            self._error = DepthModelError(f"Depth estimation failed: {e}")
        else:
            with self._lock:
                self._latest = depth

    def _run(self):
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._stopped:
                return
            with self._lock:
                frame, self._pending = self._pending, None
            if frame is not None:
                self._estimate(frame)
            self._busy = False

    def submit(self, frame):
        """Offer a frame. The frame is copied, so the caller may keep drawing on it."""
        self.submitted += 1
        if (self.submitted - 1) % self.stride != 0:
            return
        if not self.asynchronous:
            self._estimate(frame.copy())
            return
        if self._busy:
            return
        with self._lock:
            self._pending = frame.copy()
        self._busy = True
        self._wake.set()

    def latest(self):
        """The most recent depth map, or None before the first one is ready."""
        if self._error is not None:
            raise self._error
        with self._lock:
            return self._latest

    def wait_until_idle(self, timeout: float = 5.0):
        """Block until the running estimation, if any, has finished."""
        if self._thread is None:
            return
        deadline = timeout
        while self._busy and deadline > 0:
            self._thread.join(0.01)
            deadline -= 0.01

    def close(self):
        self._stopped = True
        self._wake.set()
        if self._thread is not None:
            self._thread.join()
