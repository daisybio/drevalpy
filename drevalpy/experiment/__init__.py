"""Experiment sub-module: single-run execution and result types."""

from drevalpy.types.run_result import RunResult
from drevalpy.types.trial_result import TrialResult

from .randomization import randomization
from .robustness import shuffled_splits
from .run import run

__all__ = ["RunResult", "TrialResult", "run", "randomization", "shuffled_splits"]
