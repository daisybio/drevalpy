"""Tests for :mod:`drevalpy.models._combinations`.

``combinations`` is a stub until W1-P4 lands; the result types are real.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from drevalpy import models
from drevalpy.models._combinations import Combinations, SkippedCombination, combinations


def test_skipped_combination_is_a_frozen_recipe_and_reason() -> None:
    skipped = SkippedCombination("a:b:c", "needs fingerprints")

    assert (skipped.recipe, skipped.reason) == ("a:b:c", "needs fingerprints")
    with pytest.raises(dataclasses.FrozenInstanceError):
        skipped.reason = "other"  # type: ignore[misc]


def test_combinations_result_holds_specs_and_skipped() -> None:
    skipped = SkippedCombination("a:b:c", "reason")

    result = Combinations(specs=[models.ModelSpec.from_name("NaivePredictor")], skipped=[skipped])

    assert [spec.name for spec in result.specs] == ["NaivePredictor"]
    assert result.skipped == [skipped]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.specs = []  # type: ignore[misc]


def test_signature_is_keyword_only_with_required_predictors() -> None:
    parameters = inspect.signature(combinations).parameters

    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in parameters.values())
    assert parameters["sample"].default == ()
    assert parameters["drug"].default == ()
    assert parameters["predictors"].default is inspect.Parameter.empty


def test_is_not_implemented_yet() -> None:
    with pytest.raises(NotImplementedError, match="W1-P4"):
        combinations(predictors=["elasticNet"])
