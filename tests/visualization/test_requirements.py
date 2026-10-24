"""Tests for :mod:`drevalpy.visualization.requirements`."""

from __future__ import annotations

from drevalpy.visualization.requirements import PlotRequirement


def test_declares_exactly_the_four_known_capabilities_with_distinct_values():
    assert [r.name for r in PlotRequirement] == ["MULTIPLE_MODELS", "MULTIPLE_FOLDS", "RANDOMIZATION", "ROBUSTNESS"]
    assert len({r.value for r in PlotRequirement}) == len(list(PlotRequirement))


def test_is_hashable_so_plots_can_declare_frozensets():
    requirements = frozenset({PlotRequirement.MULTIPLE_MODELS, PlotRequirement.MULTIPLE_FOLDS})

    assert PlotRequirement.MULTIPLE_MODELS in requirements
    assert PlotRequirement.RANDOMIZATION not in requirements
