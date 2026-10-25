"""Hyperparameter search spaces, public/qualified key resolution, and the exported public mapping."""

from __future__ import annotations

import re
import warnings

import optuna
import pytest

from drevalpy.models import ModelSpec
from drevalpy.models.config import from_spec
from drevalpy.models.config.model import ModelConfig
from drevalpy.models.tuning.hyperparameter_export import export_public_mapping
from drevalpy.models.tuning.hyperparameter_keys import build_ownership_index, resolve_to_qualified_mapping
from drevalpy.models.tuning.public_flat import (
    apply_public_hyperparameters_to_config,
    public_hyperparameters_from_config,
)
from drevalpy.models.tuning.search_space import (
    apply_merged_to_model_config,
    defaults_from_merged_space,
    draw_from_optuna_trial,
    extract_defaults,
    merge_model_config_spaces,
    merge_search_spaces,
    split_hyperparameters,
)
from drevalpy.models.zoo import get_zoo_config

TWO_PCA = "pca[expression]+pca[proteomics]:fingerprints:randomForest"
QUALIFIED = {
    "sample_featurizer.pca[expression].n_components": 32,
    "sample_featurizer.pca[proteomics].n_components": 16,
}


def test_merge_split_and_extract_defaults() -> None:
    merged = merge_search_spaces(
        sample_featurizer_space={"n_components": {"type": "int", "low": 4, "high": 64, "default": 8}},
        drug_featurizer_space={"n_bits": {"type": "int", "low": 64, "high": 256, "default": 128}},
        predictor_space={"alpha": {"type": "float", "low": 0.1, "high": 1.0, "default": 0.5}},
    )
    assert set(merged) == {"sample_featurizer.n_components", "drug_featurizer.n_bits", "predictor.alpha"}

    values = {"sample_featurizer.n_components": 8, "drug_featurizer.n_bits": 128, "predictor.alpha": 0.5}
    assert split_hyperparameters(values) == ({"n_components": 8}, {"n_bits": 128}, {"alpha": 0.5})
    assert split_hyperparameters({"alpha": 1.0, "l1_ratio": 0.5})[2] == {"alpha": 1.0, "l1_ratio": 0.5}

    assert extract_defaults(
        sample_featurizer_space={"n_components": {"type": "int", "default": 16}},
        predictor_space={"alpha": {"type": "float", "default": 0.1}},
    ) == {"sample_featurizer.n_components": 16, "predictor.alpha": 0.1}


def test_merge_concat_child_spaces_use_qualified_selectors() -> None:
    config = from_spec("pca[expression]+landmarkGenes:fingerprints:randomForest")
    assert isinstance(config, ModelConfig)
    merged = merge_model_config_spaces(config)
    assert any(key.startswith("sample_featurizer.pca[expression].") for key in merged)
    assert any(key.startswith("sample_featurizer.landmarkGenes.") for key in merged)
    assert any("predictor.randomForest." in key for key in merged)
    # The class-level space is what HPO tunes; it must not drift from the config merge.
    assert ModelSpec("ComboRF", config).hyperparameter_space == merged


def test_apply_merged_to_model_config() -> None:
    config = from_spec("pca[expression]:identity:randomForest")
    assert isinstance(config, ModelConfig)
    merged = defaults_from_merged_space(merge_model_config_spaces(config))

    updated = apply_merged_to_model_config(config, merged)

    assert updated.featurizer_values("sample", "pca[expression]")["n_components"] == 128
    assert all("." in key for key in updated.values)
    with pytest.raises(ValueError, match="Indexed featurizer hyperparameter keys are no longer supported"):
        apply_merged_to_model_config(config, {"sample_featurizer.pca.0.n_components": 8})


class TestDrawFromOptunaTrial:
    @pytest.mark.parametrize(
        ("spec", "is_in_range"),
        [
            pytest.param({"type": "int", "low": 10, "high": 20, "default": 15}, lambda v: 10 <= v <= 20, id="int"),
            pytest.param(
                {"type": "float", "low": 0.1, "high": 0.9, "default": 0.2},
                lambda v: 0.1 <= v <= 0.9,
                id="float",
            ),
            pytest.param(
                {"type": "float", "low": 0.001, "high": 10.0, "log": True, "default": 1.0},
                lambda v: 0.001 <= v <= 10.0,
                id="log-float",
            ),
            pytest.param(
                {"type": "categorical", "choices": ["linear", "rbf", "poly"], "default": "rbf"},
                lambda v: v in {"linear", "rbf", "poly"},
                id="categorical",
            ),
            pytest.param(42, lambda v: v == 42, id="non-mapping-is-passed-through"),
        ],
    )
    def test_each_spec_kind_draws_inside_its_own_domain(self, spec, is_in_range) -> None:
        trial = optuna.create_study().ask()

        drawn = draw_from_optuna_trial(trial, {"param": spec})

        assert is_in_range(drawn["param"])

    def test_list_valued_choices_are_drawn_whole_and_without_an_optuna_warning(self) -> None:
        choices = [[32, 16, 8, 4], [128, 64, 32], [16, 8, 4]]
        spec = {"type": "categorical", "choices": choices, "default": choices[0]}
        trial = optuna.create_study().ask()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            drawn = draw_from_optuna_trial(trial, {"units_per_layer": spec})

        assert drawn["units_per_layer"] in choices

    def test_the_qualified_key_survives_drawing(self) -> None:
        space = {
            "predictor.randomForest.n_estimators": {"type": "int", "low": 10, "high": 20, "default": 15},
            "predictor.randomForest.max_samples": {"type": "float", "low": 0.1, "high": 0.9, "default": 0.2},
        }
        trial = optuna.create_study().ask()

        drawn = draw_from_optuna_trial(trial, space)

        assert set(drawn) == set(space)


def test_public_round_trip_restores_defaults() -> None:
    config = from_spec("pca[expression]:identity:randomForest")
    public = public_hyperparameters_from_config(config)
    assert not any("." in key for key in public)
    rebuilt = apply_public_hyperparameters_to_config(config, public)
    assert rebuilt.featurizer_values("sample", "pca[expression]")["n_components"] == 128

    elastic = get_zoo_config("ElasticNet")
    assert apply_public_hyperparameters_to_config(elastic, public_hyperparameters_from_config(elastic)) is not None


def test_methylation_n_components_alias() -> None:
    config = from_spec("scaledGeneExpression+pca[methylation]:fingerprints:randomForest")
    updated = apply_public_hyperparameters_to_config(config, {"methylation_n_components": 7})
    assert updated.featurizer_values("sample", "pca[methylation]")["n_components"] == 7


def test_qualified_n_components_update_single_leaf() -> None:
    updated = apply_public_hyperparameters_to_config(from_spec(TWO_PCA), QUALIFIED)
    assert updated.featurizer_values("sample", "pca[expression]")["n_components"] == 32
    assert updated.featurizer_values("sample", "pca[proteomics]")["n_components"] == 16


def test_ambiguous_n_components_rejected_for_multi_pca_stack() -> None:
    with pytest.raises(ValueError, match="Ambiguous hyperparameter 'n_components'"):
        apply_public_hyperparameters_to_config(from_spec(TWO_PCA), {"n_components": 64})


@pytest.mark.parametrize(
    ("model", "flat"),
    [("ElasticNet", {"alpha": 0.1, "totally_unknown_key": 42}), ("MultiViewRandomForest", {"sample_views": ["x"]})],
)
def test_unknown_flat_keys_rejected(model: str, flat: dict) -> None:
    with pytest.raises(ValueError, match="Unknown hyperparameter"):
        apply_public_hyperparameters_to_config(from_spec(model), flat)


def test_tuned_config_strips_structured_keys() -> None:
    config = from_spec("pca[expression]:identity:randomForest")
    tuned = apply_merged_to_model_config(config, defaults_from_merged_space(merge_model_config_spaces(config)))
    assert "sample_featurizer.pca[expression].n_components" not in public_hyperparameters_from_config(tuned)


def test_model_spec_default_hyperparameters_are_flat() -> None:
    spec = ModelSpec("PcaOneHotRF", from_spec("pca[expression]:identity:randomForest"))
    assert not any("." in key for key in spec.default_hyperparameters)


def test_export_uses_qualified_keys_for_collisions() -> None:
    config = from_spec("pca[expression]+pca[proteomics]:fingerprints:randomForest")
    assert isinstance(config, ModelConfig)
    exported = export_public_mapping(config)
    assert "n_components" not in exported
    assert "sample_featurizer.pca[expression].n_components" in exported
    assert "sample_featurizer.pca[proteomics].n_components" in exported


def _resolve(config, mapping, reserved=frozenset()):
    return resolve_to_qualified_mapping(config, mapping, build_ownership_index(config), reserved_keys=reserved)


def test_ownership_index_lists_every_slot_of_the_stack() -> None:
    index = build_ownership_index(from_spec("pca[expression]:fingerprints:elasticNet"))

    assert {target.slot for target in index.targets} == {"predictor", "sample_featurizer", "drug_featurizer"}
    assert [t.qualified_key for t in index.short_to_targets["alpha"]] == ["predictor.elasticNet.alpha"]
    assert index.qualified_to_target["drug_featurizer.fingerprints.n_bits"].param == "n_bits"
    assert index.alias_to_qualified == {}


def test_single_owner_and_ambiguous_short_keys() -> None:
    config = from_spec(TWO_PCA)
    assert len(build_ownership_index(config).short_to_targets["n_components"]) == 2
    assert _resolve(config, QUALIFIED) == QUALIFIED
    with pytest.raises(ValueError, match="Ambiguous hyperparameter 'n_components'.*pca\\[expression\\]"):
        _resolve(config, {"n_components": 64})


def test_a_short_key_resolves_to_its_single_owner() -> None:
    assert _resolve(from_spec("pca[expression]:fingerprints:elasticNet"), {"alpha": 0.1, "radius": 3}) == {
        "predictor.elasticNet.alpha": 0.1,
        "drug_featurizer.fingerprints.radius": 3,
    }


def test_the_methylation_alias_only_applies_to_stacks_with_the_target() -> None:
    with_pca = from_spec("scaledGeneExpression+pca[methylation]:fingerprints:randomForest")
    assert build_ownership_index(with_pca).alias_to_qualified == {
        "methylation_n_components": "sample_featurizer.pca[methylation].n_components"
    }
    assert _resolve(with_pca, {"methylation_n_components": 7}) == {"sample_featurizer.pca[methylation].n_components": 7}
    with pytest.raises(ValueError, match="Unknown hyperparameter 'methylation_n_components'"):
        _resolve(from_spec("ElasticNet"), {"methylation_n_components": 7})


def test_reserved_keys_are_left_out() -> None:
    config = from_spec("ElasticNet")
    assert _resolve(config, {"alpha": 0.2, "sample_views": ["x"]}, frozenset({"sample_views"})) == {
        "predictor.elasticNet.alpha": 0.2
    }


@pytest.mark.parametrize(
    ("first", "second"),
    [("alpha", "predictor.elasticNet.alpha"), ("predictor.elasticNet.alpha", "alpha")],
    ids=["short-then-qualified", "qualified-then-short"],
)
def test_duplicate_short_and_qualified_assignments_rejected(first: str, second: str) -> None:
    with pytest.raises(ValueError, match=f"Duplicate hyperparameter assignment.*{first!r} and {second!r}"):
        _resolve(from_spec("ElasticNet"), {first: 0.2, second: 0.3})


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ("predictor.randomForest.n_estimators", "predictor 'randomForest' does not match stack predictor 'elasticNet'"),
        ("predictor.elasticNet", r"expected predictor\.<name>\.<param>"),
        ("predictor.elasticNet.not_a_knob", "for this model stack"),
        ("sample_featurizer.pca", r"expected <slot>\.<selector>\.<param>"),
        ("drug_featurizer.fingerprints.not_a_knob", "for this model stack"),
        ("sample_featurizer.pca[proteomics].n_components", "for this model stack"),
        ("sample_featurizer.pca.n_components", "for this model stack"),
        ("something.else", "for this model stack"),
        ("not_a_knob", "for this model stack"),
    ],
)
def test_unknown_keys_are_rejected_with_a_specific_message(key: str, message: str) -> None:
    config = from_spec("pca[expression]:fingerprints:elasticNet")
    with pytest.raises(ValueError, match=re.escape(f"Unknown hyperparameter {key!r}") + f".*{message}"):
        _resolve(config, {key: 1})


def test_indexed_featurizer_keys_report_the_migration_error() -> None:
    with pytest.raises(ValueError, match="no longer supported"):
        _resolve(from_spec("pca[expression]:fingerprints:elasticNet"), {"sample_featurizer.pca.0.n_components": 8})
