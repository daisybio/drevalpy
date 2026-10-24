"""Result types: RunResult, TrialResult, ModelResult, ExperimentResult, RunFailure."""

from .experiment import ExperimentResult
from .failure import ExperimentFailedError, RunFailure
from .model import ModelResult
from .run import RunResult
from .trial import TrialResult

__all__ = ["ExperimentFailedError", "ExperimentResult", "ModelResult", "RunFailure", "RunResult", "TrialResult"]
