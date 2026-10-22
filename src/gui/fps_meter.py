from collections import deque


class FpsMeter:
    def __init__(self, window_seconds=2.0):
        self.window_seconds = window_seconds
        self.reset()

    def reset(self):
        self._times = deque()
        self._frames = 0
        self._first = None
        self._last = None
        self._paused_total = 0.0
        self._paused_at = None

    @property
    def frames(self):
        return self._frames

    def tick(self, now):
        if self._first is None:
            self._first = now
        self._last = now
        self._frames += 1
        self._times.append(now)
        while len(self._times) > 2 and now - self._times[0] > self.window_seconds:
            self._times.popleft()

    def pause(self, now):
        if self._first is not None:
            self._paused_at = now

    def resume(self, now):
        if self._paused_at is not None:
            self._paused_total += now - self._paused_at
            self._paused_at = None
        self._times.clear()

    def current_fps(self):
        if len(self._times) < 2:
            return None
        span = self._times[-1] - self._times[0]
        if span <= 0:
            return None
        return (len(self._times) - 1) / span

    def average_fps(self):
        if self._frames < 2:
            return None
        active = (self._last - self._first) - self._paused_total
        if active <= 0:
            return None
        return (self._frames - 1) / active


def describe_speed(fps, source_fps):
    text = f"{fps:.1f} fps"
    if source_fps:
        text += f" ({fps / source_fps:.2f}x real time)"
    return text
