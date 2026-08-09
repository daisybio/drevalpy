"""Experiment sub-module: single-run execution and result types."""

from drevalpy.types.run_result import RunResult
from drevalpy.types.trial_result import TrialResult

from .run import run

__all__ = ["RunResult", "TrialResult", "run"]
