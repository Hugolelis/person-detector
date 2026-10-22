from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from src.reporting.chart import draw_empty_occupancy, draw_occupancy

REDRAW_INTERVAL_S = 1.0


class LiveChart(FigureCanvasQTAgg):
    def __init__(self, limit, redraw_interval=REDRAW_INTERVAL_S):
        self.figure = Figure(figsize=(4.4, 6), layout="constrained")
        super().__init__(self.figure)
        self.axes = self.figure.subplots()
        self.limit = limit
        self.redraw_interval = redraw_interval
        self.last_redraw_time = None
        self.timestamps = []
        self.counts = []
        self.redraw()

    def add_point(self, time_s, count):
        self.timestamps.append(time_s)
        self.counts.append(count)
        if (
            self.last_redraw_time is None
            or time_s - self.last_redraw_time >= self.redraw_interval
        ):
            self.redraw()

    def set_limit(self, limit):
        self.limit = limit
        self.redraw()

    def reset(self):
        self.timestamps = []
        self.counts = []
        self.last_redraw_time = None
        self.redraw()

    def redraw(self):
        if self.timestamps:
            self.last_redraw_time = self.timestamps[-1]
        self.axes.clear()
        if self.timestamps:
            draw_occupancy(self.axes, self.timestamps, self.counts, self.limit)
        else:
            draw_empty_occupancy(self.axes, self.limit)
        self.draw_idle()
