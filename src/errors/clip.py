from .base import PersonDetectorError


class ClipWriteError(PersonDetectorError):
    """Raised when a clip file cannot be opened for writing."""
