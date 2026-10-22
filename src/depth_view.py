import cv2
import numpy as np

HIGHLIGHT_COLOR = (255, 0, 255)
RANK_COLOR = (255, 255, 255)
BASE_REGION = 0.5
CENTER_REGION = 0.6


def person_depth(depth_map, box):
    """Typical depth of a person: median over the lower, central part of the box.

    The head and the sides of a box often show the background, the lower center is
    almost always the person's body. Returns None when the box is outside the map.
    """
    x1, y1, x2, y2 = box[:4]
    margin = (1 - CENTER_REGION) / 2 * (x2 - x1)
    top = y1 + (1 - BASE_REGION) * (y2 - y1)
    height, width = depth_map.shape[:2]
    left, right = max(round(x1 + margin), 0), min(round(x2 - margin) + 1, width)
    upper, lower = max(round(top), 0), min(round(y2) + 1, height)
    if right <= left or lower <= upper:
        return None
    return float(np.median(depth_map[upper:lower, left:right]))


def rank_by_closeness(depth_map, detections):
    """Detections as (index, depth) pairs, from the closest person to the farthest."""
    ranking = []
    for index, detection in enumerate(detections):
        depth = person_depth(depth_map, detection)
        if depth is not None:
            ranking.append((index, depth))
    return sorted(ranking, key=lambda item: item[1], reverse=True)


def colorize_depth(depth_map):
    """Color a 0..1 depth map: warm colors are close, dark ones are far."""
    gray = (depth_map * 255).astype("uint8")
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def confirmed_first(ranking, closest):
    """The ranking with the confirmed closest person moved to the front."""
    if closest is None:
        return ranking
    return [item for item in ranking if item[0] == closest] + [
        item for item in ranking if item[0] != closest
    ]


def draw_ranking(image, detections, ranking, closest=None):
    """Draw each ranked person's box with its position, highlighting closest."""
    thickness = max(2, image.shape[1] // 640)
    font_scale = image.shape[1] / 1536 * 1.2
    for position, (index, _depth) in enumerate(ranking, start=1):
        x1, y1, x2, y2 = (int(value) for value in detections[index][:4])
        is_closest = index == closest
        color = HIGHLIGHT_COLOR if is_closest else RANK_COLOR
        box_thickness = thickness * (2 if is_closest else 1)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness)
        label = f"#{position} closest" if is_closest else f"#{position}"
        cv2.putText(
            image,
            label,
            (x1, max(y1 - thickness * 2, int(font_scale * 20))),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
        )
    return image


def render_depth_view(depth_map, detections, ranking):
    """Depth map with the raw ranking drawn, treating its first person as closest."""
    closest = ranking[0][0] if ranking else None
    return draw_ranking(colorize_depth(depth_map), detections, ranking, closest)
