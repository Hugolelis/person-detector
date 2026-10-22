import pytest
from PySide6.QtWidgets import QApplication
from src.gui.widgets.speed_readout import SpeedReadout


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_a_file_source_shows_its_fps(app):
    readout = SpeedReadout()
    readout.set_source(live=False, fps=23.976)
    assert readout.text() == "Source video: 23.98 fps"


def test_a_camera_source_is_labelled_live(app):
    readout = SpeedReadout()
    readout.set_source(live=True, fps=30)
    assert readout.text() == "Source: webcam (live)"


def test_begin_shows_that_the_speed_is_being_measured(app):
    readout = SpeedReadout()
    readout.set_source(live=False, fps=10)
    readout.begin()
    assert readout.text().endswith("Processing: measuring...")


def test_pause_and_resume_update_the_text(app):
    readout = SpeedReadout()
    readout.begin()
    readout.pause()
    assert "paused" in readout.text()
    readout.resume()
    assert "measuring" in readout.text()


def test_finishing_without_frames_says_so(app):
    readout = SpeedReadout()
    readout.begin()
    readout.finish()
    assert "stopped before the first frame" in readout.text()
