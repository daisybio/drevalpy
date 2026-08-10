"""Result types: RunResult, TrialResult, ModelResult, ExperimentResult."""

from drevalpy.types.results.experiment import ExperimentResult
from drevalpy.types.results.model import ModelResult
from drevalpy.types.results.run import RunResult
from drevalpy.types.results.trial import TrialResult

__all__ = ["ExperimentResult", "ModelResult", "RunResult", "TrialResult"]
