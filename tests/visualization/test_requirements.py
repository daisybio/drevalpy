"""Tests for :mod:`drevalpy.visualization.requirements`.

``PlotRequirement`` is the vocabulary plots use to declare what data they need;
``ExperimentResult.satisfies`` consumes it. Only the enum itself is asserted
here - the matching logic lives with the result types and the selection logic
lives with the visualization registry.
"""

from __future__ import annotations

import pytest

from drevalpy.visualization.requirements import PlotRequirement


class TestPlotRequirement:
    def test_declares_exactly_the_four_known_capabilities(self):
        assert [r.name for r in PlotRequirement] == [
            "MULTIPLE_MODELS",
            "MULTIPLE_FOLDS",
            "RANDOMIZATION",
            "ROBUSTNESS",
        ]

    def test_values_are_distinct(self):
        assert len({r.value for r in PlotRequirement}) == len(list(PlotRequirement))

    @pytest.mark.parametrize(
        "name",
        [
            pytest.param("MULTIPLE_MODELS", id="multiple_models"),
            pytest.param("MULTIPLE_FOLDS", id="multiple_folds"),
            pytest.param("RANDOMIZATION", id="randomization"),
            pytest.param("ROBUSTNESS", id="robustness"),
        ],
    )
    def test_members_are_reachable_by_name(self, name):
        assert PlotRequirement[name].name == name

    def test_is_hashable_so_plots_can_declare_frozensets(self):
        requirements = frozenset({PlotRequirement.MULTIPLE_MODELS, PlotRequirement.MULTIPLE_FOLDS})

        assert PlotRequirement.MULTIPLE_MODELS in requirements
        assert PlotRequirement.RANDOMIZATION not in requirements

    def test_members_are_singletons(self):
        assert PlotRequirement.ROBUSTNESS is PlotRequirement["ROBUSTNESS"]
