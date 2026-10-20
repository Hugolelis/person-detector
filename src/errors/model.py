from .base import PersonDetectorError


class ModelLoadError(PersonDetectorError):
    """Raised when the detection model cannot be loaded."""
