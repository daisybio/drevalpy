"""Tests for the qualified hyperparameter keys of a model: resolving user input and checking keys and values."""

from __future__ import annotations

import re
from typing import Any

import pytest

from drevalpy.components.contracts.hyperparameter_space import admits
from drevalpy.models import ModelSpec
from drevalpy.models.config import from_recipe
from drevalpy.models.hyperparameters import component_values, key, qualify, validate
from drevalpy.models.tuning.search_space import space
from drevalpy.models.zoo import list_zoo_names

TWO_PCA = "pca[expression]+pca[proteomics]:fingerprints:randomForest"


@pytest.fixture(scope="module")
def known() -> dict[str, Any]:
    return space(from_recipe(TWO_PCA))


def test_key_and_component_values_round_trip() -> None:
    qualified = key("sample_featurizer", "pca[methylation]", "n_components")

    assert qualified == "sample_featurizer.pca[methylation].n_components"
    values = {qualified: 4, "predictor.elasticNet.alpha": 1.0}
    assert component_values(values, "sample_featurizer", "pca[methylation]") == {"n_components": 4}
    assert component_values(values, "predictor", "elasticNet") == {"alpha": 1.0}
    assert component_values(values, "drug_featurizer", "fingerprints") == {}


def test_short_and_qualified_keys_resolve_to_the_same_mapping(known: dict[str, Any]) -> None:
    qualified = {
        "predictor.randomForest.n_estimators": 60,
        "sample_featurizer.pca[expression].n_components": 16,
    }
    assert qualify(known, qualified) == qualified
    assert qualify(known, {"n_estimators": 60, "sample_featurizer.pca[expression].n_components": 16}) == qualified
    assert qualify(known, {}) == {}


def test_ambiguous_unknown_and_duplicate_keys_raise(known: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="Ambiguous hyperparameter 'n_components'"):
        qualify(known, {"n_components": 4})
    with pytest.raises(ValueError, match="Unknown hyperparameter 'nope'"):
        qualify(known, {"nope": 1})
    with pytest.raises(ValueError, match="Duplicate hyperparameter assignment"):
        qualify(known, {"n_estimators": 60, "predictor.randomForest.n_estimators": 70})


def test_validate_accepts_only_qualified_keys_of_the_space(known: dict[str, Any]) -> None:
    validate(known, {})
    validate(known, {"predictor.randomForest.n_estimators": 10})


@pytest.mark.parametrize(
    "name",
    [
        "n_estimators",
        "predictor.randomForest.not_a_knob",
        "predictor.elasticNet.alpha",
        "sample_featurizer.pca.n_components",
        "drug_featurizer.fingerprints[smiles].radius",
        "sample_featurizer.concatFeaturizers.n_components",
    ],
    ids=["unqualified", "unknown-param", "another-predictor", "missing-view", "unexpected-view", "concat-parent"],
)
def test_validate_rejects_a_key_the_model_does_not_own(known: dict[str, Any], name: str) -> None:
    with pytest.raises(ValueError, match="Unknown hyperparameter"):
        validate(known, {name: 1})


OUTSIDE = [
    ("predictor.randomForest.n_estimators", 5, "int from 50 to 200"),
    ("predictor.randomForest.max_samples", 1.0, "float from 0.1 to 0.5"),
    ("drug_featurizer.fingerprints.n_bits", 256, "a power of two from 512 to 2048"),
    ("drug_featurizer.fingerprints.n_bits", 1000, "a power of two from 512 to 2048"),
    ("drug_featurizer.fingerprints.use_counts", "yes", "one of [True, False]"),
]
OUTSIDE_IDS = ["int", "float", "pow2-below", "pow2-not-a-power", "categorical"]


@pytest.mark.parametrize(("name", "value", "described"), OUTSIDE, ids=OUTSIDE_IDS)
def test_qualify_rejects_a_value_outside_the_space(
    known: dict[str, Any], name: str, value: Any, described: str
) -> None:
    message = re.escape(
        f"Hyperparameter {name!r} = {value!r} is outside its search space ({described}). Widen the space with "
        "hyperparameter_space in the model config, or pass check_values=False to with_hyperparameters or "
        "DRPModel to use it anyway."
    )
    with pytest.raises(ValueError, match=message):
        qualify(known, {name: value})
    with pytest.raises(ValueError, match=message):
        qualify(known, {name.rsplit(".", 1)[1]: value})


@pytest.mark.parametrize(("name", "value", "described"), OUTSIDE, ids=OUTSIDE_IDS)
def test_check_values_false_and_validate_take_a_value_outside_the_space(
    known: dict[str, Any], name: str, value: Any, described: str
) -> None:
    assert qualify(known, {name.rsplit(".", 1)[1]: value}, check_values=False) == {name: value}
    validate(known, {name: value})


def test_check_values_false_still_rejects_unknown_keys(known: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="Unknown hyperparameter 'nope'"):
        qualify(known, {"nope": 1}, check_values=False)


def test_the_message_states_a_widened_pow2_range_in_powers_of_two() -> None:
    known = {"drug_featurizer.fingerprints.n_bits": {"type": "pow2", "low": 8, "high": 11, "default": 256}}

    with pytest.raises(ValueError, match=r"\(a power of two from 256 to 2048\)"):
        qualify(known, {"n_bits": 128})


def test_a_fixed_entry_accepts_any_value(known: dict[str, Any]) -> None:
    assert known["predictor.randomForest.n_jobs"]["type"] == "fixed"

    assert qualify(known, {"n_jobs": "anything"}) == {"predictor.randomForest.n_jobs": "anything"}


@pytest.mark.parametrize("name", list_zoo_names(include_external=False))
def test_every_zoo_default_lies_in_its_space(name: str) -> None:
    spec = ModelSpec.from_name(name)

    outside = [key for key, value in spec.hyperparameters.items() if not admits(spec.hyperparameter_space[key], value)]
    assert outside == []
    if name == "EnsembleMF":
        n_bits = "drug_featurizer.fingerprints.n_bits"
        assert spec.hyperparameters[n_bits] == 256
        assert admits(spec.hyperparameter_space[n_bits], 256)
