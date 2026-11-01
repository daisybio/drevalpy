"""Model naming and construction: ``ModelSpec`` and ``combinations``."""

from __future__ import annotations

import dataclasses
import inspect
import re

import pytest

from drevalpy import models
from drevalpy.errors import InvalidInputError, UnknownNameError
from drevalpy.models import ModelSpec
from drevalpy.models._combinations import Combinations, SkippedCombination, combinations
from drevalpy.models.spec import UnknownModelError
from drevalpy.models.zoo import get_zoo_config


def test_from_name_returns_reusable_named_recipe() -> None:
    spec = ModelSpec.from_name("ElasticNet")

    assert spec.name == "ElasticNet"
    assert spec.config.predictor.name == "elasticNet"
    assert spec.hyperparameters["predictor.elasticNet.alpha"] == 1.0
    assert all(key.count(".") >= 2 for key in spec.hyperparameters)
    assert list(spec.hyperparameters) == list(spec.hyperparameter_space)
    with pytest.raises(TypeError):
        spec.hyperparameters["predictor.elasticNet.alpha"] = 999  # type: ignore[index]
    assert ModelSpec.from_name("NaivePredictor").hyperparameter_space == {}
    assert ModelSpec.from_name("NaivePredictor").hyperparameters == {}


def test_with_hyperparameters_accepts_short_and_qualified_keys() -> None:
    spec = ModelSpec.from_name("ElasticNet")

    short = spec.with_hyperparameters({"alpha": 0.2})
    qualified = spec.with_hyperparameters({"predictor.elasticNet.alpha": 0.2})

    assert short == qualified
    assert short.hyperparameters["predictor.elasticNet.alpha"] == 0.2
    assert (
        short.hyperparameters["predictor.elasticNet.l1_ratio"] == spec.hyperparameters["predictor.elasticNet.l1_ratio"]
    )
    assert spec.hyperparameters["predictor.elasticNet.alpha"] == 1.0
    assert short.with_hyperparameters({"l1_ratio": 0.5}).hyperparameters["predictor.elasticNet.alpha"] == 0.2
    with pytest.raises(ValueError, match="Unknown hyperparameter 'nope'"):
        spec.with_hyperparameters({"nope": 1})


def test_hyperparameters_must_be_qualified_keys_of_the_config() -> None:
    config = get_zoo_config("ElasticNet")

    assert (
        ModelSpec("custom", config, {"predictor.elasticNet.alpha": 0.3}).hyperparameters["predictor.elasticNet.alpha"]
        == 0.3
    )
    with pytest.raises(ValueError, match="Unknown hyperparameter 'alpha'"):
        ModelSpec("custom", config, {"alpha": 0.3})


def test_dump_and_load_round_trip() -> None:
    spec = ModelSpec.from_name("MultiViewRandomForest").with_hyperparameters({"n_estimators": 8}, check_values=False)

    dumped = spec.dump()

    assert dumped["hyperparameters"]["predictor.randomForest.n_estimators"] == 8
    assert ModelSpec.load(dumped) == spec


def test_check_values_false_takes_values_outside_the_space_but_not_unknown_keys() -> None:
    spec = ModelSpec.from_name("RandomForest")

    with pytest.raises(ValueError, match="is outside its search space"):
        spec.with_hyperparameters({"n_estimators": 5})
    with pytest.raises(ValueError, match="Unknown hyperparameter 'nope'"):
        spec.with_hyperparameters({"nope": 1}, check_values=False)
    small = spec.with_hyperparameters({"n_estimators": 5}, check_values=False)

    assert small.hyperparameters["predictor.randomForest.n_estimators"] == 5
    assert ModelSpec(small.name, small.config, small.hyperparameters) == small
    assert ModelSpec.load(small.dump()) == small


def test_model_id_is_the_recipe_of_the_components() -> None:
    assert ModelSpec.from_name("pca[gene_expression]:fingerprints:elasticNet").model_id == (
        "pca[gene_expression]:fingerprints:elasticNet"
    )
    assert ModelSpec.from_name("NaivePredictor").model_id == "naiveMean"


def test_spec_shares_its_deep_frozen_model_config() -> None:
    config = get_zoo_config("ElasticNet")
    spec = ModelSpec("custom", config)

    assert spec.config is config
    with pytest.raises(TypeError):
        spec.config.sample_featurizer.options["gene_list"] = "other"  # type: ignore[index, union-attr]


def test_with_hyperparameters_keeps_the_config_and_the_computed_space() -> None:
    spec = ModelSpec.from_name("ElasticNet")

    derived = spec.with_hyperparameters({"alpha": 0.2})

    assert derived.config is spec.config
    assert derived.hyperparameter_space is spec.hyperparameter_space


def test_invalid_constructor_arguments_are_rejected() -> None:
    for name in ("", "  "):
        with pytest.raises(ValueError, match="non-empty"):
            ModelSpec(name, get_zoo_config("ElasticNet"))
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
        ("view[gene_expression]:fingerprints:naiveTissueMean", "needs sample block"),
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
        sample=["view[gene_expression]", "pca[gene_expression]"],
        drug=["fingerprints"],
        predictors=["elasticNet", "naiveTissueMean"],
    )

    assert [spec.name for spec in result.specs] == [
        "view[gene_expression]:fingerprints:elasticNet",
        "pca[gene_expression]:fingerprints:elasticNet",
    ]
    assert [item.recipe for item in result.skipped] == [
        "view[gene_expression]:fingerprints:naiveTissueMean",
        "pca[gene_expression]:fingerprints:naiveTissueMean",
    ]
    assert result.skipped[0].reason == (
        "Predictor 'naiveTissueMean' needs sample block(s) ['tissue'] but the sample featurizer 'view' "
        "emits [gene_expression]"
    )
    assert all("\n" not in item.reason for item in result.skipped)


def test_concat_atom_supplies_the_block_a_predictor_needs() -> None:
    result = combinations(sample=["view[gene_expression]+tissue"], drug=["constant"], predictors=["naiveTissueMean"])

    assert [spec.name for spec in result.specs] == ["view[gene_expression]+tissue:constant:naiveTissueMean"]
    assert result.skipped == []


def test_empty_sample_and_drug_leave_only_the_predictor() -> None:
    result = combinations(predictors=["naiveMean", "elasticNet"])

    assert [spec.name for spec in result.specs] == ["naiveMean"]
    assert [item.recipe for item in result.skipped] == ["elasticNet"]
    assert "requires a sample_featurizer" in result.skipped[0].reason


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
    assert result.skipped[0].reason.startswith("Unknown sample featurizer 'nope'")


def test_no_valid_combination_raises_with_every_reason() -> None:
    with pytest.raises(InvalidInputError, match="No valid combination among 2") as caught:
        combinations(
            sample=["view[gene_expression]", "pca[gene_expression]"],
            drug=["fingerprints"],
            predictors=["naiveTissueMean"],
        )

    message = str(caught.value)
    for sample in ("view", "pca"):
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
