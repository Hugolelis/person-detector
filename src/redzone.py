import cv2
import numpy as np


class RedZone:
    def __init__(self, points: list[list[int]]):
        self.polygon = np.array(points, dtype=np.int32)

    def contains(self, point: tuple[float, float]) -> bool:
        return cv2.pointPolygonTest(self.polygon, point, False) >= 0

    def count_people(self, detections: list[tuple]) -> int:
        count = 0
        for x1, _y1, x2, y2, _conf in detections:
            foot_point = ((x1 + x2) / 2, y2)
            if self.contains(foot_point):
                count += 1
        return count
