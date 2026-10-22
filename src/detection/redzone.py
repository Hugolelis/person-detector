import cv2
import numpy as np

DEFAULT_MIN_OVERLAP_PERCENT = 30.0


class RedZone:
    def __init__(
        self,
        points: list[list[int]],
        min_overlap_percent: float = DEFAULT_MIN_OVERLAP_PERCENT,
    ):
        """A person counts when min_overlap_percent of their box is inside the zone.

        0 counts anyone whose box touches the zone, 100 needs the whole box inside.
        """
        self.polygon = np.array(points, dtype=np.int32)
        self.min_overlap = min_overlap_percent / 100

        self._origin = self.polygon.min(axis=0)
        size = self.polygon.max(axis=0) - self._origin + 1
        mask = np.zeros((size[1], size[0]), dtype=np.uint8)
        cv2.fillPoly(mask, [self.polygon - self._origin], 1)
        # Summed-area table: the zone pixels inside any rectangle in four lookups.
        self._inside = cv2.integral(mask)

    def contains(self, point: tuple[float, float]) -> bool:
        return cv2.pointPolygonTest(self.polygon, point, False) >= 0

    def box_overlap(self, box: tuple) -> float:
        """Fraction (0 to 1) of the box area that lies inside the zone."""
        x1, y1, x2, y2 = (round(value) for value in box[:4])
        area = (x2 - x1) * (y2 - y1)
        if area <= 0:
            return 0.0

        height, width = self._inside.shape[0] - 1, self._inside.shape[1] - 1
        left = min(max(x1 - int(self._origin[0]), 0), width)
        right = min(max(x2 - int(self._origin[0]), 0), width)
        top = min(max(y1 - int(self._origin[1]), 0), height)
        bottom = min(max(y2 - int(self._origin[1]), 0), height)

        table = self._inside
        pixels = (
            table[bottom, right]
            - table[top, right]
            - table[bottom, left]
            + table[top, left]
        )
        return float(pixels) / area

    def count_people(self, detections: list[tuple]) -> int:
        count = 0
        for detection in detections:
            overlap = self.box_overlap(detection)
            if overlap > 0 and overlap >= self.min_overlap:
                count += 1
        return count
