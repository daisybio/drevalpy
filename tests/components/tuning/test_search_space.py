"""Tests for internal hyperparameter search-space helpers."""

import pytest

import drevalpy.components.register_builtins as register_builtins
from drevalpy.components.tuning.search_space import (
    apply_merged_to_model_config,
    defaults_from_merged_space,
    extract_defaults,
    merge_model_config_spaces,
    merge_search_spaces,
    split_hyperparameters,
)
from drevalpy.models.config import model_config_from_spec


def test_merge_all_three_spaces() -> None:
    merged = merge_search_spaces(
        cell_line_featurizer_space={"n_components": {"type": "int", "low": 4, "high": 64, "default": 8}},
        drug_featurizer_space={"n_bits": {"type": "int", "low": 64, "high": 256, "default": 128}},
        predictor_space={"alpha": {"type": "float", "low": 0.1, "high": 1.0, "default": 0.5}},
    )
    assert set(merged) == {
        "cell_line_featurizer.n_components",
        "drug_featurizer.n_bits",
        "predictor.alpha",
    }


def test_split_hyperparameters_inverts_merge() -> None:
    merged = {
        "cell_line_featurizer.n_components": 8,
        "drug_featurizer.n_bits": 128,
        "predictor.alpha": 0.5,
    }
    cell_line_hp, drug_hp, predictor_hp = split_hyperparameters(merged)
    assert cell_line_hp == {"n_components": 8}
    assert drug_hp == {"n_bits": 128}
    assert predictor_hp == {"alpha": 0.5}


def test_split_predictor_only_fallback() -> None:
    merged = {"alpha": 1.0, "l1_ratio": 0.5}
    _, _, predictor_hp = split_hyperparameters(merged)
    assert predictor_hp == {"alpha": 1.0, "l1_ratio": 0.5}


def test_merge_concat_child_spaces_use_qualified_selectors() -> None:
    register_builtins.register_builtin_components()
    config = model_config_from_spec("pca[expression]+landmarkGenes:fingerprints:randomForest")
    merged = merge_model_config_spaces(config)
    pca_keys = [key for key in merged if key.startswith("cell_line_featurizer.pca[expression].")]
    assert pca_keys
    assert any(key.startswith("cell_line_featurizer.landmarkGenes.") for key in merged)
    assert any("predictor.randomForest." in key for key in merged)


def test_merge_same_name_different_views_get_distinct_keys() -> None:
    register_builtins.register_builtin_components()
    config = model_config_from_spec("pca[expression]+pca[proteomics]:fingerprints:randomForest")
    merged = merge_model_config_spaces(config)
    assert any(key.startswith("cell_line_featurizer.pca[expression].") for key in merged)
    assert any(key.startswith("cell_line_featurizer.pca[proteomics].") for key in merged)


def test_apply_rejects_indexed_featurizer_keys() -> None:
    register_builtins.register_builtin_components()
    config = model_config_from_spec("pca[expression]:identity:randomForest")
    with pytest.raises(
        ValueError,
        match="Indexed featurizer hyperparameter keys are no longer supported",
    ):
        apply_merged_to_model_config(
            config,
            {"cell_line_featurizer.pca.0.n_components": 8},
        )


def test_apply_merged_to_model_config_strips_featurizer_prefix() -> None:
    register_builtins.register_builtin_components()
    config = model_config_from_spec("pca[expression]:identity:randomForest")
    merged = defaults_from_merged_space(merge_model_config_spaces(config))
    updated = apply_merged_to_model_config(config, merged)
    assert updated.cell_line_featurizer is not None
    assert updated.cell_line_featurizer.hyperparameters == {"n_components": 128}
    assert not any("." in key for key in updated.cell_line_featurizer.hyperparameters)


def test_extract_defaults() -> None:
    defaults = extract_defaults(
        cell_line_featurizer_space={"n_components": {"type": "int", "default": 16}},
        predictor_space={"alpha": {"type": "float", "default": 0.1}},
    )
    assert defaults == {
        "cell_line_featurizer.n_components": 16,
        "predictor.alpha": 0.1,
    }
