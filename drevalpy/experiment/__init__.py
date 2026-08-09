"""Experiment sub-module: single-run execution and result types."""

from .run import run
from .run_result import RunResult
from .trial_result import TrialResult

__all__ = ["RunResult", "TrialResult", "run"]
