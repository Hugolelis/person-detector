import cv2
import numpy as np
from src.video_reader import VideoReader

WINDOW_NAME = "Calibration"
DISPLAY_WIDTH = 1280
POINT_COLOR = (0, 255, 255)
LINE_COLOR = (0, 0, 255)
ENTER_KEY = 13
ESC_KEY = 27


def _draw(display, points, scale):
    canvas = display.copy()
    shown = [(round(x * scale), round(y * scale)) for x, y in points]
    for point in shown:
        cv2.circle(canvas, point, 6, POINT_COLOR, -1)
    if len(shown) >= 2:
        polygon = np.array(shown, dtype=np.int32)
        cv2.polylines(canvas, [polygon], len(shown) >= 3, LINE_COLOR, 2)
    return canvas


def calibrate_roi(video_path):
    with VideoReader(video_path) as reader:
        frame = next(reader.frames())

    scale = DISPLAY_WIDTH / frame.shape[1]
    display = cv2.resize(frame, None, fx=scale, fy=scale)
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((round(x / scale), round(y / scale)))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    print("Left click: add | Right click or u: undo | Enter: confirm | Esc/q: cancel")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    try:
        while True:
            cv2.imshow(WINDOW_NAME, _draw(display, points, scale))
            key = cv2.waitKey(20) & 0xFF

            if key == ord("u") and points:
                points.pop()
            elif key == ENTER_KEY and len(points) >= 3:
                return points
            elif key in (ESC_KEY, ord("q")):
                return None

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return None
    finally:
        cv2.destroyAllWindows()
