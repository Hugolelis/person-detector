from src.enums.violation_event import ViolationEvent


class ViolationDetector:
    def __init__(
        self,
        person_limit: int,
        frames_to_confirm_start: int,
        frames_to_confirm_end: int,
        cooldown_frames: int,
    ):
        self.person_limit = person_limit
        self.frames_to_confirm_start = frames_to_confirm_start
        self.frames_to_confirm_end = frames_to_confirm_end
        self.cooldown_frames = cooldown_frames

        self.in_violation = False
        self.frames_above = 0
        self.frames_below = 0
        self.cooldown_remaining = 0

    def update(self, count: int) -> ViolationEvent | None:
        if not self.in_violation:
            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1
                return None

            if count > self.person_limit:
                self.frames_above += 1
                if self.frames_above >= self.frames_to_confirm_start:
                    self.in_violation = True
                    self.frames_above = 0
                    return ViolationEvent.STARTED
            else:
                self.frames_above = 0
        else:
            if count > self.person_limit:
                self.frames_below = 0
            else:
                self.frames_below += 1
                if self.frames_below >= self.frames_to_confirm_end:
                    self.in_violation = False
                    self.frames_below = 0
                    self.cooldown_remaining = self.cooldown_frames
                    return ViolationEvent.ENDED

        return None
