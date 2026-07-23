"""Errors raised when predictor state cannot be restored."""


class PredictorStateError(RuntimeError):
    """Raised when ``set_state`` receives invalid or incomplete predictor state."""
