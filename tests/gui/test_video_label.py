from src.gui.widgets.video_label import widget_to_frame

FRAME_SIZE = (3840, 2160)
PIXMAP_SIZE = (640, 360)
WIDGET_SIZE = (800, 500)


def test_click_at_pixmap_center_maps_to_frame_center():
    x = WIDGET_SIZE[0] / 2
    y = WIDGET_SIZE[1] / 2
    assert widget_to_frame(x, y, WIDGET_SIZE, PIXMAP_SIZE, FRAME_SIZE) == (1920, 1080)


def test_click_at_pixmap_top_left_maps_to_frame_origin():
    offset_x = (WIDGET_SIZE[0] - PIXMAP_SIZE[0]) / 2
    offset_y = (WIDGET_SIZE[1] - PIXMAP_SIZE[1]) / 2
    assert widget_to_frame(
        offset_x, offset_y, WIDGET_SIZE, PIXMAP_SIZE, FRAME_SIZE
    ) == (0, 0)


def test_click_in_the_empty_margin_is_ignored():
    assert widget_to_frame(5, 5, WIDGET_SIZE, PIXMAP_SIZE, FRAME_SIZE) is None


def test_click_scales_by_the_frame_to_pixmap_ratio():
    offset_x = (WIDGET_SIZE[0] - PIXMAP_SIZE[0]) / 2
    offset_y = (WIDGET_SIZE[1] - PIXMAP_SIZE[1]) / 2
    result = widget_to_frame(
        offset_x + 100, offset_y + 50, WIDGET_SIZE, PIXMAP_SIZE, FRAME_SIZE
    )
    assert result == (600, 300)
