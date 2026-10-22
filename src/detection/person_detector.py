import torch
from src.errors.model import ModelLoadError
from ultralytics import YOLO

PERSON_CLASS_ID = 0
DEFAULT_THREADS = 4


class PersonDetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.4,
        threads: int = DEFAULT_THREADS,
    ):
        self.conf = conf
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise ModelLoadError(f"Could not load model: {model_path}") from e
        torch.set_num_threads(threads)

    def detect(self, frame):
        """Detect people in a frame.

        Returns:
            list[tuple]: one (x1, y1, x2, y2, confidence) tuple per person.
        """
        results = self.model(
            frame, classes=[PERSON_CLASS_ID], conf=self.conf, verbose=False
        )
        boxes = results[0].boxes

        detections = []
        for xyxy, conf in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), strict=True):
            x1, y1, x2, y2 = xyxy
            detections.append((x1, y1, x2, y2, conf))
        return detections
