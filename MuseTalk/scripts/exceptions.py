class MuseTalkError(Exception):
    """Base class for all MuseTalk exceptions."""


class FFmpegExecutionError(MuseTalkError):
    """Raised when an FFmpeg command fails."""


class AvatarProcessingError(MuseTalkError):
    """Raised when an avatar frame processing error occurs."""


class InferenceError(MuseTalkError):
    """Raised when an inference pipeline error occurs."""


class ModelNotFoundError(MuseTalkError):
    """Raised when an inference pipeline error occurs."""


class FFmpegNotFoundError(Exception):
    """Raised when FFmpeg binary cannot be located or executed."""
    pass

