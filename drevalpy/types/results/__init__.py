"""Result types: RunResult, TrialResult, ModelResult, ExperimentResult."""

from .experiment import ExperimentResult
from .model import ModelResult
from .run import RunResult
from .trial import TrialResult

__all__ = ["ExperimentResult", "ModelResult", "RunResult", "TrialResult"]
