from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from src.chart import draw_empty_occupancy, draw_occupancy


class LiveChart(FigureCanvasQTAgg):
    def __init__(self, limit):
        self.figure = Figure(figsize=(4.4, 6), layout="constrained")
        super().__init__(self.figure)
        self.axes = self.figure.subplots()
        self.limit = limit
        self.timestamps = []
        self.counts = []
        self.redraw()

    def add_point(self, time_s, count):
        self.timestamps.append(time_s)
        self.counts.append(count)
        self.redraw()

    def set_limit(self, limit):
        self.limit = limit
        self.redraw()

    def reset(self):
        self.timestamps = []
        self.counts = []
        self.redraw()

    def redraw(self):
        self.axes.clear()
        if self.timestamps:
            draw_occupancy(self.axes, self.timestamps, self.counts, self.limit)
        else:
            draw_empty_occupancy(self.axes, self.limit)
        self.draw_idle()
