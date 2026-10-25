"""Model naming and construction: ``ModelSpec``, the factory, ``combinations`` and the hyperparameter key grammar."""

from __future__ import annotations

import dataclasses
import inspect
import re

import pytest

from drevalpy import models
from drevalpy.errors import InvalidInputError, UnknownNameError
from drevalpy.models import ModelSpec
from drevalpy.models._combinations import Combinations, SkippedCombination, combinations
from drevalpy.models._hp_key_grammar import (
    DRUG_SLOT,
    FEATURIZER_SLOTS,
    PREDICTOR_SLOT,
    REGISTRY_TO_SLOT,
    SAMPLE_SLOT,
    SLOT_TO_REGISTRY,
    featurizer_prefix,
    is_featurizer_slot_key,
    predictor_prefix,
    reject_indexed_featurizer_key,
    split_predictor_key,
    split_prefixed_key,
)
from drevalpy.models.config import ResolvedModelConfig, from_spec
from drevalpy.models.factory import model_config_for_name
from drevalpy.models.spec import UnknownModelError
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.types.enums.prediction_mode import PredictionMode


def test_from_name_returns_reusable_named_recipe() -> None:
    spec = ModelSpec.from_name("ElasticNet")

    assert spec.name == "ElasticNet"
    assert spec.config.predictor.name == "elasticNet"
    assert spec.resolve().predictor_values()["alpha"] == spec.default_hyperparameters["alpha"]
    assert all("." not in key for key in spec.default_hyperparameters)
    spec.default_hyperparameters["alpha"] = 999
    assert spec.default_hyperparameters["alpha"] != 999
    assert ModelSpec.from_name("NaivePredictor").hyperparameter_space == {}


def test_spec_snapshots_its_model_config() -> None:
    config = from_spec("ElasticNet")
    spec = ModelSpec("custom", config)

    assert spec.config is not config
    assert spec.config.model_dump(mode="python") == config.model_dump(mode="python")


def test_invalid_constructor_arguments_are_rejected() -> None:
    for name in ("", "  "):
        with pytest.raises(ValueError, match="non-empty"):
            ModelSpec(name, from_spec("ElasticNet"))
    with pytest.raises(TypeError, match="ModelConfig"):
        ModelSpec("bad", object())  # type: ignore[arg-type]


def test_from_name_reads_a_recipe_and_names_it_by_the_string_as_given() -> None:
    recipe = "pca[gene_expression]:fingerprints:elasticNet"

    spec = ModelSpec.from_name(recipe)

    assert spec.name == recipe
    assert spec.config.predictor.name == "elasticNet"
    assert spec.config.sample_featurizer.name == "pca"
    assert spec.config.sample_featurizer.view == "gene_expression"


def test_from_name_prefers_a_zoo_name_and_accepts_a_feature_free_token() -> None:
    assert ModelSpec.from_name("NaivePredictor").config.predictor.name == "naiveMean"
    assert ModelSpec.from_name("naiveMean").name == "naiveMean"


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("NotAModel", "Unknown model spec 'NotAModel'"),
        ("pca:elasticNet", "requires an explicit view"),
        ("raw[gene_expression]:fingerprints:naiveTissueMean", "needs sample block"),
        ("a:b:c:d", "Malformed model recipe"),
    ],
)
def test_from_name_error_names_both_readings_and_the_recipe_reason(name: str, reason: str) -> None:
    with pytest.raises(
        UnknownNameError, match=f"{re.escape(repr(name))} is not a zoo name and not a valid recipe: .*{reason}"
    ) as caught:
        ModelSpec.from_name(name)

    assert isinstance(caught.value, UnknownModelError)
    assert caught.value.name == name
    assert reason in caught.value.detail
    assert "Invalid model config" not in caught.value.detail


def test_model_config_for_name_matches_spec_resolution() -> None:
    flat = {"n_estimators": 8}
    via_factory = model_config_for_name("MultiViewRandomForest", flat)
    via_spec = ModelSpec.from_name("MultiViewRandomForest").resolve(flat)
    assert via_factory.predictor_values()["n_estimators"] == via_spec.predictor_values()["n_estimators"] == 8


def test_model_config_for_name_forwards_prediction_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_predictor("elasticNet"), "supported_modes", frozenset(PredictionMode))
    config = model_config_for_name("ElasticNet", prediction_mode=PredictionMode.CLASSIFICATION)
    assert not isinstance(config, ResolvedModelConfig)
    assert config.prediction_mode == PredictionMode.CLASSIFICATION


def test_result_types_are_frozen_and_the_signature_is_keyword_only() -> None:
    skipped = SkippedCombination("a:b:c", "needs fingerprints")
    result = Combinations(specs=[models.ModelSpec.from_name("NaivePredictor")], skipped=[skipped])

    assert (skipped.recipe, skipped.reason) == ("a:b:c", "needs fingerprints")
    assert [spec.name for spec in result.specs] == ["NaivePredictor"]
    assert result.skipped == [skipped]
    with pytest.raises(dataclasses.FrozenInstanceError):
        skipped.reason = "other"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.specs = []  # type: ignore[misc]

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
    for sample in ("raw", "pca"):
        assert f"{sample}[gene_expression]:fingerprints:naiveTissueMean: Predictor 'naiveTissueMean' needs" in message


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sample": ["pca[gene_expression]"], "predictors": []}, "at least one predictor"),
        ({"drug": ["fingerprints"], "predictors": ["elasticNet"]}, "drug featurizers need a sample featurizer"),
        ({"sample": "pca[gene_expression]"}, "sample must be a list of recipe tokens"),
        ({"sample": ["a:b"]}, "sample entries must be non-empty recipe tokens"),
        ({"sample": ["  "]}, "sample entries must be non-empty recipe tokens"),
        ({"sample": [3]}, "sample entries must be non-empty recipe tokens"),
    ],
)
def test_invalid_arguments_are_rejected(kwargs: dict, match: str) -> None:
    with pytest.raises(InvalidInputError, match=match):
        combinations(**{"predictors": ["elasticNet"], **kwargs})


def test_slot_constants_are_the_wire_format() -> None:
    assert (SAMPLE_SLOT, DRUG_SLOT, PREDICTOR_SLOT) == ("sample_featurizer", "drug_featurizer", "predictor")
    assert FEATURIZER_SLOTS == (SAMPLE_SLOT, DRUG_SLOT)
    assert REGISTRY_TO_SLOT == {"sample": SAMPLE_SLOT, "drug": DRUG_SLOT}
    assert SLOT_TO_REGISTRY == {slot: registry for registry, slot in REGISTRY_TO_SLOT.items()}


@pytest.mark.parametrize(
    ("registry", "selector", "expected"),
    [
        ("sample", "pca", "sample_featurizer.pca.n_components"),
        ("sample", "pca[methylation]", "sample_featurizer.pca[methylation].n_components"),
        ("drug", "fingerprints", "drug_featurizer.fingerprints.n_components"),
    ],
)
def test_prefix_builders(registry: str, selector: str, expected: str) -> None:
    assert featurizer_prefix(registry, selector, "n_components") == expected
    assert predictor_prefix("elasticNet", "alpha") == "predictor.elasticNet.alpha"
    with pytest.raises(KeyError):
        featurizer_prefix("proteome", "pca", "n_components")


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("sample_featurizer.pca.n_components", True),
        ("drug_featurizer.fingerprints.radius", True),
        ("predictor.elasticNet.alpha", False),
        ("n_components", False),
        ("sample_featurizer", False),
    ],
)
def test_is_featurizer_slot_key(key: str, expected: bool) -> None:
    assert is_featurizer_slot_key(key) is expected


@pytest.mark.parametrize(
    "key",
    [
        "sample_featurizer.pca.0.n_components",
        "drug_featurizer.fingerprints.1.radius",
        "sample_featurizer.pca.12.n_components",
        "sample_featurizer.pca.0.nested.param",
    ],
)
def test_indexed_featurizer_keys_are_rejected(key: str) -> None:
    with pytest.raises(ValueError, match="no longer supported"):
        reject_indexed_featurizer_key(key)
    with pytest.raises(ValueError, match=r"sample_featurizer\.pca\[<view>\]\.n_components"):
        reject_indexed_featurizer_key("sample_featurizer.pca.0.n_components")


@pytest.mark.parametrize(
    "key",
    [
        "sample_featurizer.pca.n_components",
        "sample_featurizer.pca[gene_expression].n_components",
        "sample_featurizer.pca.view.n_components",
        "predictor.elasticNet.0.alpha",
        "sample_featurizer.pca.0",
    ],
)
def test_other_keys_are_not_rejected_as_indexed(key: str) -> None:
    assert reject_indexed_featurizer_key(key) is None


@pytest.mark.parametrize(
    ("registry", "selector", "param"),
    [
        ("sample", "pca", "n_components"),
        ("sample", "pca[methylation]", "n_components"),
        ("drug", "fingerprints", "radius"),
        ("drug", "fingerprints", "nested.param"),
    ],
)
def test_split_prefixed_key_round_trips_a_built_key(registry: str, selector: str, param: str) -> None:
    assert split_prefixed_key(featurizer_prefix(registry, selector, param)) == (registry, selector, param)


@pytest.mark.parametrize("key", ["predictor.elasticNet.alpha", "n_components", "sample_featurizer.pca"])
def test_split_prefixed_key_returns_none_for_a_non_featurizer_key(key: str) -> None:
    assert split_prefixed_key(key) is None


def test_split_prefixed_key_propagates_the_indexed_key_rejection() -> None:
    with pytest.raises(ValueError, match="no longer supported"):
        split_prefixed_key("sample_featurizer.pca.0.n_components")


@pytest.mark.parametrize(("name", "param"), [("elasticNet", "alpha"), ("randomForest", "nested.param")])
def test_split_predictor_key_round_trips_a_built_key(name: str, param: str) -> None:
    assert split_predictor_key(predictor_prefix(name, param)) == (name, param)


@pytest.mark.parametrize("key", ["sample_featurizer.pca.n_components", "predictor.elasticNet", "alpha"])
def test_split_predictor_key_returns_none_for_a_non_predictor_key(key: str) -> None:
    assert split_predictor_key(key) is None
