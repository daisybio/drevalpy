"""Tests for the public surface of the results package."""

from __future__ import annotations

from drevalpy.types import results
from tests._barrel_surface import DeclaredSurface


class TestResultsSurface(DeclaredSurface):
    barrel = results
    origins = {
        "ExperimentFailedError": "drevalpy.types.results.failure",
        "ExperimentResult": "drevalpy.types.results.experiment",
        "ModelResult": "drevalpy.types.results.model",
        "RunResult": "drevalpy.types.results.run",
    }
