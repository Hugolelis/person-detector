import cv2
import numpy as np
from src.errors.depth import DepthModelError

DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEFAULT_INPUT_SIZE = 266
PATCH_SIZE = 14
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _multiple_of_patch(value: float) -> int:
    return max(PATCH_SIZE, round(value / PATCH_SIZE) * PATCH_SIZE)


def prepare_input(frame, input_size: int = DEFAULT_INPUT_SIZE):
    """Turn a BGR frame into the (3, H, W) float array the model expects.

    The short side becomes input_size and the long side keeps the aspect ratio. Both
    are multiples of the model patch size.
    """
    height, width = frame.shape[:2]
    short = _multiple_of_patch(input_size)
    if height <= width:
        new_height, new_width = short, _multiple_of_patch(short * width / height)
    else:
        new_height, new_width = _multiple_of_patch(short * height / width), short
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


def normalize_depth(depth, low_percentile: float = 2, high_percentile: float = 98):
    """Scale a relative depth map to 0..1, where 1 is the closest to the camera."""
    low, high = np.percentile(depth, [low_percentile, high_percentile])
    if high <= low:
        return np.zeros_like(depth, dtype=np.float32)
    return np.clip((depth - low) / (high - low), 0, 1).astype(np.float32)


class DepthEstimator:
    def __init__(
        self,
        model_name: str = DEFAULT_DEPTH_MODEL,
        input_size: int = DEFAULT_INPUT_SIZE,
    ):
        """Load a Hugging Face depth model. Needs the optional transformers package."""
        try:
            import torch
            from transformers import AutoModelForDepthEstimation
        except ImportError as e:
            raise DepthModelError(
                "The depth view needs the 'transformers' package. "
                "Install it with: poetry add transformers"
            ) from e
        try:
            self.model = AutoModelForDepthEstimation.from_pretrained(model_name).eval()
        except Exception as e:
            raise DepthModelError(
                f"Could not load the depth model: {model_name}"
            ) from e
        self.torch = torch
        self.input_size = input_size

    def estimate(self, frame):
        """Return a float32 map, the size of frame, with 1 where it is closest."""
        batch = self.torch.from_numpy(prepare_input(frame, self.input_size))[None]
        with self.torch.inference_mode():
            predicted = self.model(pixel_values=batch).predicted_depth[0].numpy()
        height, width = frame.shape[:2]
        depth = cv2.resize(predicted, (width, height), interpolation=cv2.INTER_LINEAR)
        return normalize_depth(depth)
