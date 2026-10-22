from PySide6.QtCore import Qt

MIN_POINTS = 3


class ZoneEditor:
    """State of the zone being drawn: the points clicked so far, no widgets."""

    def __init__(self):
        self.active = False
        self.points = []

    def begin(self):
        self.active = True
        self.points = []

    def end(self):
        self.active = False

    def click(self, x, y, button):
        """Left click adds a point, right click removes the last one."""
        if button == Qt.MouseButton.LeftButton:
            self.points.append((x, y))
        elif button == Qt.MouseButton.RightButton and self.points:
            self.points.pop()

    def undo(self) -> bool:
        if not self.points:
            return False
        self.points.pop()
        return True

    @property
    def can_save(self) -> bool:
        return len(self.points) >= MIN_POINTS

    def roi(self):
        return [list(point) for point in self.points]
