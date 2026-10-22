from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel


def widget_to_frame(x, y, widget_size, pixmap_size, frame_size):
    offset_x = (widget_size[0] - pixmap_size[0]) / 2
    offset_y = (widget_size[1] - pixmap_size[1]) / 2
    inside_x = x - offset_x
    inside_y = y - offset_y
    if not (0 <= inside_x < pixmap_size[0] and 0 <= inside_y < pixmap_size[1]):
        return None
    return (
        round(inside_x * frame_size[0] / pixmap_size[0]),
        round(inside_y * frame_size[1] / pixmap_size[1]),
    )


class VideoLabel(QLabel):
    clicked = Signal(int, int, object)

    def __init__(self, text=""):
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.click_frame_size = None

    def mousePressEvent(self, event):
        pixmap = self.pixmap()
        if self.click_frame_size is None or pixmap is None or pixmap.isNull():
            return
        position = widget_to_frame(
            event.position().x(),
            event.position().y(),
            (self.width(), self.height()),
            (pixmap.width(), pixmap.height()),
            self.click_frame_size,
        )
        if position is not None:
            self.clicked.emit(position[0], position[1], event.button())
