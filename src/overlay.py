import cv2
import numpy as np

ZONE_COLOR = (0, 0, 255)
BOX_COLOR = (0, 255, 0)
ALERT_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)
POINT_COLOR = (0, 255, 255)


def _line_thickness(frame):
    return max(2, frame.shape[1] // 640)


def draw_zone(frame, zone):
    thickness = _line_thickness(frame)
    cv2.polylines(
        frame, [zone.polygon], isClosed=True, color=ZONE_COLOR, thickness=thickness
    )


def draw_zone_preview(frame, points):
    thickness = _line_thickness(frame)
    for point in points:
        cv2.circle(frame, tuple(point), thickness * 3, POINT_COLOR, -1)
    if len(points) >= 2:
        polygon = np.array(points, dtype=np.int32)
        cv2.polylines(frame, [polygon], len(points) >= 3, ZONE_COLOR, thickness)


def draw_detections(frame, detections):
    thickness = _line_thickness(frame)
    font_scale = frame.shape[1] / 1536
    for x1, y1, x2, y2, conf in detections:
        top_left = (int(x1), int(y1))
        bottom_right = (int(x2), int(y2))
        cv2.rectangle(frame, top_left, bottom_right, BOX_COLOR, thickness)
        cv2.putText(
            frame,
            f"{conf:.2f}",
            (top_left[0], top_left[1] - thickness * 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            BOX_COLOR,
            thickness,
        )


def draw_counter(frame, count, limit):
    thickness = _line_thickness(frame)
    font_scale = frame.shape[1] / 1536 * 1.5
    color = ALERT_COLOR if count > limit else TEXT_COLOR
    cv2.putText(
        frame,
        f"{count}/{limit}",
        (thickness * 8, thickness * 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
    )


def draw_alert(frame):
    thickness = _line_thickness(frame)
    height, width = frame.shape[:2]
    font_scale = frame.shape[1] / 1536 * 1.5
    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), ALERT_COLOR, thickness * 4)
    cv2.putText(
        frame,
        "VIOLATION",
        (thickness * 8, thickness * 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        ALERT_COLOR,
        thickness,
    )


def draw_overlay(frame, detections, zone, count, limit, in_violation):
    draw_zone(frame, zone)
    draw_detections(frame, detections)
    draw_counter(frame, count, limit)
    if in_violation:
        draw_alert(frame)
    return frame
