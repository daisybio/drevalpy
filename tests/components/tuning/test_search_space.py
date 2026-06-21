"""Tests for internal hyperparameter search-space helpers."""

from drevalpy.components.tuning.search_space import (
    extract_defaults,
    merge_search_spaces,
    split_hyperparameters,
)


def test_merge_all_three_spaces() -> None:
    merged = merge_search_spaces(
        cell_line_featurizer_space={"n_components": {"type": "int", "low": 4, "high": 64, "default": 8}},
        drug_featurizer_space={"n_bits": {"type": "int", "low": 64, "high": 256, "default": 128}},
        predictor_space={"alpha": {"type": "float", "low": 0.1, "high": 1.0, "default": 0.5}},
    )
    assert set(merged) == {
        "featurizer.cell_line.n_components",
        "featurizer.drug.n_bits",
        "predictor.alpha",
    }


def test_split_hyperparameters_inverts_merge() -> None:
    merged = {
        "featurizer.cell_line.n_components": 8,
        "featurizer.drug.n_bits": 128,
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


def test_extract_defaults() -> None:
    defaults = extract_defaults(
        cell_line_featurizer_space={"n_components": {"type": "int", "default": 16}},
        predictor_space={"alpha": {"type": "float", "default": 0.1}},
    )
    assert defaults == {
        "featurizer.cell_line.n_components": 16,
        "predictor.alpha": 0.1,
    }
