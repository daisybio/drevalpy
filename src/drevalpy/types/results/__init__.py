"""Results of runs, models and experiments, and the error raised when runs failed.

The names below are the ones user code imports to load results or catch the error;
:class:`~drevalpy.types.results.trial.TrialResult` and
:class:`~drevalpy.types.results.failure.RunFailure` are only handed out, so they stay
in their defining modules.
"""

from .experiment import ExperimentResult
from .failure import ExperimentFailedError
from .model import ModelResult
from .run import RunResult

__all__ = ["ExperimentFailedError", "ExperimentResult", "ModelResult", "RunResult"]
