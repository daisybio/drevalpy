"""Tests for the ChemBERTa drug featurizer; the embedding path is the ``network``-marked test."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drevalpy.components.featurizers.drug.chemberta import ChemBertaFeaturizer
from tests._import_shims import block_imports
from tests.conftest import MockFeatureSource

_HIDDEN = torch.tensor([[[1.0, 2.0], [3.0, 8.0], [5.0, 2.0]]])


def test_hyperparameter_space_exposes_pooling_and_max_length() -> None:
    assert set(ChemBertaFeaturizer.get_hyperparameter_space()) == {"pooling", "max_length"}


@pytest.mark.parametrize(
    ("pooling", "expected"),
    [("cls", [1.0, 2.0]), ("max", [5.0, 8.0]), ("mean", [3.0, 4.0]), ("not-a-strategy", [3.0, 4.0])],
)
def test_pool_applies_the_configured_strategy_and_falls_back_to_mean(pooling: str, expected: list[float]) -> None:
    np.testing.assert_allclose(ChemBertaFeaturizer(pooling=pooling)._pool(_HIDDEN), expected)


def test_compute_from_source_without_smiles_raises() -> None:
    source = MockFeatureSource(features={"d1": {}})

    with pytest.raises(ValueError, match="no SMILES available"):
        ChemBertaFeaturizer()._compute_from_source(source, np.array(["d1"]))


def test_transform_blocks_are_named_chemberta() -> None:
    source = MockFeatureSource(
        features={"d1": {"chemberta": np.array([0.1, 0.2])}},
        meta_info={"chemberta": ["e1", "e2"]},
    )
    ids = np.array(["d1"], dtype=str)
    blocks = ChemBertaFeaturizer().fit(source, entity_ids=ids).transform_blocks(source, ids)

    assert set(blocks) == {"chemberta"}
    assert blocks["chemberta"].feature_names == ("e1", "e2")


def test_load_chemberta_reports_missing_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    from drevalpy.components.featurizers.drug.chemberta import load_chemberta

    load_chemberta.cache_clear()
    block_imports(monkeypatch, "transformers")

    with pytest.raises(ImportError, match="transformers and torch are required"):
        load_chemberta()

    load_chemberta.cache_clear()


@pytest.mark.network
def test_compute_from_source_embeds_dataset_smiles(synthetic_dataset) -> None:
    from drevalpy.types.data.feature_source import DrugFeatureSource

    source = DrugFeatureSource(synthetic_dataset, synthetic_dataset.drug_ids)

    matrix = ChemBertaFeaturizer(max_length=64)._compute_from_source(source, synthetic_dataset.drug_ids[:2])

    assert matrix.shape[0] == 2
    assert matrix.dtype == np.float32
