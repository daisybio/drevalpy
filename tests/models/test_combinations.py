"""Tests for :mod:`drevalpy.models._combinations`."""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from drevalpy import models
from drevalpy.errors import InvalidInputError
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


def test_grid_keeps_valid_products_and_reports_the_skipped_with_reasons() -> None:
    result = combinations(
        sample=["raw[gene_expression]", "pca[gene_expression]"],
        drug=["fingerprints"],
        predictors=["elasticNet", "naiveTissueMean"],
    )

    assert [spec.name for spec in result.specs] == [
        "raw[gene_expression]:fingerprints:elasticNet",
        "pca[gene_expression]:fingerprints:elasticNet",
    ]
    assert [item.recipe for item in result.skipped] == [
        "raw[gene_expression]:fingerprints:naiveTissueMean",
        "pca[gene_expression]:fingerprints:naiveTissueMean",
    ]
    assert result.skipped[0].reason == (
        "Predictor 'naiveTissueMean' needs sample block(s) ['tissue'] but the sample featurizer 'raw' "
        "emits [gene_expression]"
    )
    assert all("\n" not in item.reason for item in result.skipped)


def test_concat_atom_supplies_the_block_a_predictor_needs() -> None:
    result = combinations(sample=["raw[gene_expression]+tissue"], drug=["constant"], predictors=["naiveTissueMean"])

    assert [spec.name for spec in result.specs] == ["raw[gene_expression]+tissue:constant:naiveTissueMean"]
    assert result.skipped == []


def test_empty_sample_and_drug_leave_only_the_predictor() -> None:
    result = combinations(predictors=["naiveMean", "elasticNet"])

    assert [spec.name for spec in result.specs] == ["naiveMean"]
    assert [item.recipe for item in result.skipped] == ["elasticNet"]
    assert "requires featurizers" in result.skipped[0].reason


def test_empty_drug_omits_the_slot() -> None:
    with pytest.raises(ValueError, match=r"pca\[gene_expression\]:elasticNet: .*requires a drug_featurizer"):
        combinations(sample=["pca[gene_expression]"], predictors=["elasticNet"])


def test_identical_recipes_are_built_once() -> None:
    result = combinations(
        sample=["pca[gene_expression]", " pca[gene_expression]"],
        drug=["fingerprints", "fingerprints"],
        predictors=["elasticNet", "elasticNet"],
    )

    assert [spec.name for spec in result.specs] == ["pca[gene_expression]:fingerprints:elasticNet"]


def test_unknown_component_is_skipped_with_the_registry_message() -> None:
    result = combinations(sample=["pca[gene_expression]", "nope"], drug=["fingerprints"], predictors=["elasticNet"])

    assert [spec.name for spec in result.specs] == ["pca[gene_expression]:fingerprints:elasticNet"]
    assert result.skipped == [
        SkippedCombination("nope:fingerprints:elasticNet", result.skipped[0].reason),
    ]
    assert result.skipped[0].reason.startswith("Unknown Sample featurizer: 'nope'")


def test_no_valid_combination_raises_with_every_reason() -> None:
    with pytest.raises(InvalidInputError, match="No valid combination among 2") as caught:
        combinations(
            sample=["raw[gene_expression]", "pca[gene_expression]"],
            drug=["fingerprints"],
            predictors=["naiveTissueMean"],
        )

    message = str(caught.value)
    assert (
        "raw[gene_expression]:fingerprints:naiveTissueMean: Predictor 'naiveTissueMean' needs sample block" in message
    )
    assert (
        "pca[gene_expression]:fingerprints:naiveTissueMean: Predictor 'naiveTissueMean' needs sample block" in message
    )


def test_no_predictors_is_an_error() -> None:
    with pytest.raises(InvalidInputError, match="at least one predictor"):
        combinations(sample=["pca[gene_expression]"], predictors=[])


def test_drug_without_sample_is_an_error() -> None:
    with pytest.raises(InvalidInputError, match="drug featurizers need a sample featurizer"):
        combinations(drug=["fingerprints"], predictors=["elasticNet"])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sample": "pca[gene_expression]"}, "sample must be a list of recipe tokens"),
        ({"sample": ["a:b"]}, "sample entries must be non-empty recipe tokens"),
        ({"sample": ["  "]}, "sample entries must be non-empty recipe tokens"),
        ({"sample": [3]}, "sample entries must be non-empty recipe tokens"),
    ],
)
def test_malformed_axes_are_rejected(kwargs: dict, match: str) -> None:
    with pytest.raises(InvalidInputError, match=match):
        combinations(predictors=["elasticNet"], **kwargs)
