"""Tests for landmark gene featurizers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from upath import UPath

from drevalpy.components.featurizers.cell_line import gene_lists
from drevalpy.components.featurizers.cell_line.gene_lists import gene_names_from_list_csv, resolve_gene_list_path
from drevalpy.components.featurizers.cell_line.landmark import (
    LandmarkGenesFeaturizer,
    LandmarkGenesReducedFeaturizer,
)
from tests.components.featurizers.cell_line._helpers import PRECOMPUTED, precomputed_source
from tests.conftest import MockFeatureSource


def _features() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "cl1": {"gene_expression": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)},
            "cl2": {"gene_expression": np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)},
        },
        meta_info={"gene_expression": ["A", "B", "C", "D"]},
    )


def test_landmark_uses_symbol_column_and_persists_state() -> None:
    symbols = gene_names_from_list_csv(resolve_gene_list_path("landmark_genes"))[:2]
    features = MockFeatureSource(
        features={
            "cl1": {"gene_expression": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
            "cl2": {"gene_expression": np.array([3.0, 2.0, 1.0], dtype=np.float32)},
        },
        meta_info={"gene_expression": [*symbols, "NOT_A_GENE"]},
    )
    featurizer = LandmarkGenesFeaturizer(standardize=True)
    ids = np.array(["cl1", "cl2"])
    featurizer.fit(features, entity_ids=ids)
    assert featurizer.output_dim == 2
    matrix = featurizer.transform(features, ids)
    assert matrix.shape == (2, 2)

    restored = LandmarkGenesFeaturizer()
    restored.set_state(featurizer.get_state())
    assert restored.output_dim == 2
    np.testing.assert_allclose(restored.transform(features, ids), matrix)


def test_landmark_reduced_uses_package_gene_list() -> None:
    featurizer = LandmarkGenesReducedFeaturizer(standardize=False)
    symbols = gene_names_from_list_csv(resolve_gene_list_path("landmark_genes_reduced"))[:3]
    features = MockFeatureSource(
        features={
            "cl1": {"gene_expression": np.arange(len(symbols) + 1, dtype=np.float32)},
        },
        meta_info={"gene_expression": [*symbols, "NOT_A_GENE"]},
    )
    featurizer.fit(features, entity_ids=np.array(["cl1"]))
    assert featurizer.output_dim == 3


def test_landmark_fails_clearly_on_bad_column(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pd.DataFrame({"other": ["A"]}).to_csv(tmp_path / "landmark_genes.csv", index=False)
    monkeypatch.setattr(gene_lists, "GENE_LISTS_DIR", UPath(tmp_path))
    featurizer = LandmarkGenesFeaturizer()
    with pytest.raises(ValueError, match="recognized gene-name column"):
        featurizer.fit(_features(), entity_ids=np.array(["cl1"]))


def test_landmark_requires_feature_names_on_the_view() -> None:
    features = MockFeatureSource(
        features={"cl1": {"gene_expression": np.array([1.0, 2.0], dtype=np.float32)}},
    )

    with pytest.raises(ValueError, match="no feature names for view"):
        LandmarkGenesFeaturizer().fit(features, entity_ids=np.array(["cl1"]))


def test_landmark_requires_at_least_one_matching_gene() -> None:
    with pytest.raises(ValueError, match="matched view"):
        LandmarkGenesFeaturizer().fit(_features(), entity_ids=np.array(["cl1"]))


def test_landmark_minmax_scaling_bounds_output_to_the_unit_interval() -> None:
    symbols = gene_names_from_list_csv(resolve_gene_list_path("landmark_genes"))[:2]
    features = MockFeatureSource(
        features={
            "cl1": {"gene_expression": np.array([1.0, 5.0], dtype=np.float32)},
            "cl2": {"gene_expression": np.array([9.0, 2.0], dtype=np.float32)},
        },
        meta_info={"gene_expression": list(symbols)},
    )
    ids = np.array(["cl1", "cl2"])
    featurizer = LandmarkGenesFeaturizer(standardize=True, minmax_scale=True).fit(features, entity_ids=ids)

    matrix = featurizer.transform(features, ids)

    assert matrix.min() >= 0.0
    assert matrix.max() <= 1.0


def test_landmark_without_standardization_keeps_raw_arcsinh_values() -> None:
    symbols = gene_names_from_list_csv(resolve_gene_list_path("landmark_genes"))[:2]
    features = MockFeatureSource(
        features={"cl1": {"gene_expression": np.array([0.0, 1.0], dtype=np.float32)}},
        meta_info={"gene_expression": list(symbols)},
    )
    ids = np.array(["cl1"])
    featurizer = LandmarkGenesFeaturizer(standardize=False, arcsinh=True).fit(features, entity_ids=ids)

    matrix = featurizer.transform(features, ids)

    np.testing.assert_allclose(matrix, np.arcsinh([[0.0, 1.0]]), rtol=1e-6)


def test_landmark_prefers_a_precomputed_variant() -> None:
    source = precomputed_source(LandmarkGenesFeaturizer)
    ids = source.identifiers
    featurizer = LandmarkGenesFeaturizer()

    featurizer.fit(source, entity_ids=ids)

    assert featurizer.output_dim == PRECOMPUTED.shape[1]
    np.testing.assert_allclose(featurizer.transform(source, ids), PRECOMPUTED)


def test_landmark_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        LandmarkGenesFeaturizer()._transform(_features(), np.array(["cl1"]))


def test_landmark_transform_blocks_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        LandmarkGenesFeaturizer()._transform_blocks(_features(), np.array(["cl1"]))


def test_landmark_state_is_empty_before_fit() -> None:
    assert LandmarkGenesFeaturizer().get_state() == {}


def test_landmark_hyperparameter_space_exposes_the_two_scaling_flags() -> None:
    assert set(LandmarkGenesFeaturizer.get_hyperparameter_space()) == {"standardize", "minmax_scale"}


def test_landmark_blocks_carry_the_selected_gene_names() -> None:
    symbols = gene_names_from_list_csv(resolve_gene_list_path("landmark_genes"))[:2]
    features = MockFeatureSource(
        features={"cl1": {"gene_expression": np.array([1.0, 2.0, 3.0], dtype=np.float32)}},
        meta_info={"gene_expression": [*symbols, "NOT_A_GENE"]},
    )
    ids = np.array(["cl1"])
    featurizer = LandmarkGenesFeaturizer(standardize=False).fit(features, entity_ids=ids)

    blocks = featurizer.transform_blocks(features, ids)

    assert set(blocks) == {"gene_expression"}
    assert blocks["gene_expression"].feature_names == tuple(symbols)


def test_landmark_set_state_derives_output_dim_from_gene_indices() -> None:
    featurizer = LandmarkGenesFeaturizer()

    featurizer.set_state({"gene_indices": [0, 1, 2], "output_dim": None, "fitted": True})

    assert featurizer.output_dim == 3


def test_landmark_blocks_drop_gene_names_on_a_source_without_metadata() -> None:
    """Fit needs feature names, but a later source is free not to expose any."""
    symbols = gene_names_from_list_csv(resolve_gene_list_path("landmark_genes"))[:2]
    named = MockFeatureSource(
        features={"cl1": {"gene_expression": np.array([1.0, 2.0], dtype=np.float32)}},
        meta_info={"gene_expression": list(symbols)},
    )
    unnamed = MockFeatureSource(
        features={"cl1": {"gene_expression": np.array([1.0, 2.0], dtype=np.float32)}},
    )
    ids = np.array(["cl1"])
    featurizer = LandmarkGenesFeaturizer(standardize=False).fit(named, entity_ids=ids)

    blocks = featurizer.transform_blocks(unnamed, ids)

    assert blocks["gene_expression"].feature_names is None
