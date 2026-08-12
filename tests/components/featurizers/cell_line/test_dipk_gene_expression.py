"""Tests for the DIPK gene-expression featurizer."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers.cell_line.dipk_gene_expression import (
    DIPKGeneExpressionFeaturizer,
    GeneExpressionEncoder,
)
from tests.components.featurizers.cell_line._helpers import PRECOMPUTED, precomputed_source
from tests.conftest import MockFeatureSource


def _features() -> MockFeatureSource:
    return MockFeatureSource(
        {f"cl{i}": {"gene_expression": np.arange(3, dtype=np.float32) + i} for i in range(3)},
        meta_info={"gene_expression": ["a", "b", "c"]},
    )


def test_dipk_gene_expression_round_trips_state(monkeypatch) -> None:
    features = _features()
    monkeypatch.setattr(
        "drevalpy.components.featurizers.cell_line.dipk_gene_expression.train_gene_expession_autoencoder",
        lambda train, validation, epochs: GeneExpressionEncoder(train.shape[1]),
    )
    pair_expanded_ids = np.array(["cl0", "cl1", "cl0"])
    pair_expanded_es_ids = np.array(["cl2", "cl2"])
    featurizer = DIPKGeneExpressionFeaturizer(epochs_autoencoder=1).fit(
        features, pair_expanded_ids=pair_expanded_ids, pair_expanded_es_ids=pair_expanded_es_ids
    )
    matrix = featurizer.transform(features, np.array(["cl0", "cl2"]))
    assert matrix.shape == (2, 512)
    restored = DIPKGeneExpressionFeaturizer()
    restored.set_state(featurizer.get_state())
    np.testing.assert_allclose(restored.transform(features, np.array(["cl0", "cl2"])), matrix)


def test_dipk_gene_expression_requires_pair_expanded_ids() -> None:
    with pytest.raises(ValueError, match="requires pair_expanded_ids"):
        DIPKGeneExpressionFeaturizer().fit(_features())


def test_dipk_gene_expression_requires_non_empty_early_stopping_ids() -> None:
    with pytest.raises(ValueError, match="non-empty train and early-stopping IDs"):
        DIPKGeneExpressionFeaturizer().fit(_features(), pair_expanded_ids=np.array(["cl0"]))


def test_dipk_gene_expression_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        DIPKGeneExpressionFeaturizer()._transform(_features(), np.array(["cl0"]))


def test_dipk_gene_expression_output_dim_is_zero_before_fit() -> None:
    assert DIPKGeneExpressionFeaturizer().output_dim == 0


def test_dipk_gene_expression_state_is_empty_before_fit() -> None:
    assert DIPKGeneExpressionFeaturizer().get_state() == {}


def test_dipk_gene_expression_set_state_ignores_a_malformed_payload() -> None:
    featurizer = DIPKGeneExpressionFeaturizer()

    featurizer.set_state({"encoder_state": "not-bytes", "input_dim": 3})

    assert featurizer.output_dim == 0


def test_dipk_gene_expression_hyperparameter_space_exposes_the_epoch_count() -> None:
    assert set(DIPKGeneExpressionFeaturizer.get_hyperparameter_space()) == {"epochs_autoencoder"}


def test_dipk_gene_expression_prefers_a_precomputed_variant() -> None:
    source = precomputed_source(DIPKGeneExpressionFeaturizer)
    ids = source.identifiers
    featurizer = DIPKGeneExpressionFeaturizer()

    featurizer.fit(source, pair_expanded_ids=ids)

    np.testing.assert_allclose(featurizer.transform(source, ids), PRECOMPUTED)


def test_dipk_gene_expression_blocks_use_the_precomputed_variant() -> None:
    source = precomputed_source(DIPKGeneExpressionFeaturizer)
    ids = source.identifiers
    featurizer = DIPKGeneExpressionFeaturizer().fit(source, pair_expanded_ids=ids)

    blocks = featurizer.transform_blocks(source, ids)

    assert set(blocks) == {"gene_expression"}
    np.testing.assert_allclose(blocks["gene_expression"].values, PRECOMPUTED)
