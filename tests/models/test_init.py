"""Public ``drevalpy.models`` surface; origins are deliberately unpinned."""

from __future__ import annotations

import inspect

from drevalpy import models
from tests._barrel_surface import DeclaredSurface

PROMISED_EXPORTS = (
    "ModelSpec",
    "DRPModel",
    "load_model",
    "Combinations",
    "SkippedCombination",
    "combinations",
    "get_response_transformation",
)


class TestModelsSurface(DeclaredSurface):
    barrel = models
    unpinned_names = PROMISED_EXPORTS
    callable_names = ("load_model", "combinations", "get_response_transformation")


def test_drp_model_is_exported_as_a_class() -> None:
    assert inspect.isclass(models.DRPModel)
