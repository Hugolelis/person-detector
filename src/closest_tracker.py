DEFAULT_CONFIRM_FRAMES = 5
DEFAULT_MIN_IOU = 0.3


def iou(box_a, box_b) -> float:
    """Intersection over union of two (x1, y1, x2, y2, ...) boxes, from 0 to 1."""
    left, top = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    right, bottom = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class ClosestTracker:
    """Keeps the closest person steady: a change must last confirm_frames frames.

    Detections carry no identity, so the confirmed person is followed from frame to
    frame by the overlap of their boxes.
    """

    def __init__(
        self,
        confirm_frames: int = DEFAULT_CONFIRM_FRAMES,
        min_iou: float = DEFAULT_MIN_IOU,
    ):
        self.confirm_frames = max(1, confirm_frames)
        self.min_iou = min_iou
        self.confirmed = None
        self.missing_frames = 0
        self.candidate = None
        self.candidate_frames = 0

    def _find(self, box, detections):
        """Index of the detection that overlaps box the most, if enough."""
        best_index, best_overlap = None, self.min_iou
        for index, detection in enumerate(detections):
            overlap = iou(box, detection)
            if overlap >= best_overlap:
                best_index, best_overlap = index, overlap
        return best_index

    def _follow_confirmed(self, detections):
        if self.confirmed is None:
            return None
        index = self._find(self.confirmed, detections)
        if index is None:
            self.missing_frames += 1
            if self.missing_frames >= self.confirm_frames:
                self.confirmed = None
            return None
        self.missing_frames = 0
        self.confirmed = detections[index]
        return index

    def update(self, detections, ranking):
        """Return the index of the confirmed closest person, or None.

        ranking is the (index, depth) list from the closest person to the farthest.
        """
        confirmed_index = self._follow_confirmed(detections)
        if not ranking:
            self.candidate, self.candidate_frames = None, 0
            return confirmed_index

        best_index = ranking[0][0]
        if confirmed_index == best_index:
            self.candidate, self.candidate_frames = None, 0
            return confirmed_index

        best_box = detections[best_index]
        if self.candidate is not None and iou(self.candidate, best_box) >= self.min_iou:
            self.candidate_frames += 1
        else:
            self.candidate_frames = 1
        self.candidate = best_box

        if self.candidate_frames >= self.confirm_frames:
            self.confirmed, self.missing_frames = best_box, 0
            self.candidate, self.candidate_frames = None, 0
            return best_index
        return confirmed_index
