import cv2
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy


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
        # The image must never decide the label size, or it would fight the layout.
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.click_frame_size = None
        self._source = None

    def show_image(self, pixmap):
        """Show pixmap fitted to the label and keep it fitted when the label changes."""
        self._source = pixmap
        self._fit()

    def show_frame(self, frame):
        """Show an OpenCV (BGR) frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(
            rgb.data, width, height, channels * width, QImage.Format.Format_RGB888
        )
        self.show_image(QPixmap.fromImage(image))

    def _fit(self):
        if self._source is None or self.width() < 2 or self.height() < 2:
            return
        self.setPixmap(
            self._source.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def clear(self):
        self._source = None
        super().clear()

    def setText(self, text):
        self._source = None
        super().setText(text)

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
