"""Tests for the proteomics median-centering transformer, which must stay an sklearn estimator."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers.sample._proteomics_transformer import (
    ProteomicsMedianCenterAndImputeTransformer,
)

_MATRIX = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])


@pytest.mark.parametrize(("n_features", "kept"), [(1, [0, 1]), (3, [0, 1, 2])])
def test_fit_keeps_complete_features_then_falls_back_to_the_n_most_complete(n_features: int, kept: list[int]) -> None:
    transformer = ProteomicsMedianCenterAndImputeTransformer(feature_threshold=1.0, n_features=n_features)

    assert transformer.fit(_MATRIX) is transformer
    assert sorted(transformer.protein_indices.tolist()) == kept


def test_median_centering_uses_the_mean_of_the_per_row_medians() -> None:
    transformer = ProteomicsMedianCenterAndImputeTransformer(feature_threshold=1.0, n_features=1).fit(_MATRIX)

    np.testing.assert_allclose(transformer.mean_median, 4.5)  # medians over columns 0 and 1: 1.5, 4.5, 7.5
    (row,) = transformer.transform(np.array([[4.0, 5.0, 6.0]]))
    np.testing.assert_allclose(row, [4.0, 5.0])


def test_imputation_fills_missing_values_reproducibly() -> None:
    transformer = ProteomicsMedianCenterAndImputeTransformer(feature_threshold=0.0, n_features=3).fit(_MATRIX)

    (first,) = transformer.transform(np.array([[1.0, np.nan, 3.0]]))
    (second,) = transformer.transform(np.array([[1.0, np.nan, 3.0]]))

    assert not np.isnan(first).any()
    np.testing.assert_allclose(first, second)


def test_the_transformer_satisfies_the_sklearn_estimator_protocol_and_old_import_path() -> None:
    from sklearn.base import clone

    from drevalpy.components.featurizers.sample import normalized_proteomics

    params = ProteomicsMedianCenterAndImputeTransformer(n_features=7).get_params()
    assert params["n_features"] == 7
    assert set(params) == {
        "feature_threshold",
        "imputation_seed",
        "n_features",
        "normalization_downshift",
        "normalization_width",
    }
    cloned = clone(ProteomicsMedianCenterAndImputeTransformer(feature_threshold=0.5))
    assert cloned.feature_threshold == 0.5
    assert cloned.protein_indices.size == 0
    assert (
        normalized_proteomics.ProteomicsMedianCenterAndImputeTransformer is ProteomicsMedianCenterAndImputeTransformer
    )
