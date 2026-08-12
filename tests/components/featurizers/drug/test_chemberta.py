"""Tests for the ChemBERTa drug featurizer.

Mirrors :mod:`drevalpy.components.featurizers.drug.chemberta`. Only
``load_chemberta`` needs the mirrored weight download, so the pooling strategies
are tested directly against a bare torch tensor and marked as offline; the
end-to-end embedding path is left to the ``network``-marked smoke test.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drevalpy.components.featurizers.drug.chemberta import ChemBertaFeaturizer
from tests.conftest import MockFeatureSource

_HIDDEN = torch.tensor([[[1.0, 2.0], [3.0, 8.0], [5.0, 2.0]]])


def test_hyperparameter_space_exposes_pooling_and_max_length() -> None:
    assert set(ChemBertaFeaturizer.get_hyperparameter_space()) == {"pooling", "max_length"}


@pytest.mark.parametrize(
    ("pooling", "expected"),
    [
        pytest.param("cls", [1.0, 2.0], id="cls-takes-the-first-token"),
        pytest.param("max", [5.0, 8.0], id="max-over-tokens"),
        pytest.param("mean", [3.0, 4.0], id="mean-over-tokens"),
    ],
)
def test_pool_applies_the_configured_strategy(pooling: str, expected: list[float]) -> None:
    featurizer = ChemBertaFeaturizer(pooling=pooling)

    pooled = featurizer._pool(_HIDDEN)

    np.testing.assert_allclose(pooled, expected)


def test_pool_falls_back_to_mean_for_an_unknown_strategy() -> None:
    featurizer = ChemBertaFeaturizer(pooling="not-a-strategy")

    np.testing.assert_allclose(featurizer._pool(_HIDDEN), [3.0, 4.0])


def test_compute_from_source_without_smiles_raises() -> None:
    featurizer = ChemBertaFeaturizer()
    source = MockFeatureSource(features={"d1": {}})

    with pytest.raises(ValueError, match="no SMILES available"):
        featurizer._compute_from_source(source, np.array(["d1"]))


def test_transform_blocks_are_named_chemberta() -> None:
    source = MockFeatureSource(
        features={"d1": {"chemberta": np.array([0.1, 0.2])}},
        meta_info={"chemberta": ["e1", "e2"]},
    )
    ids = np.array(["d1"], dtype=str)
    featurizer = ChemBertaFeaturizer().fit(source, entity_ids=ids)

    blocks = featurizer.transform_blocks(source, ids)

    assert set(blocks) == {"chemberta"}
    assert blocks["chemberta"].feature_names == ("e1", "e2")


def test_compute_from_source_reports_missing_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    from drevalpy.components.featurizers.drug.chemberta import load_chemberta

    load_chemberta.cache_clear()
    real_import = builtins.__import__

    def _fail_on_transformers(name, *args, **kwargs):
        if name.startswith("transformers"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail_on_transformers)

    with pytest.raises(ImportError, match="transformers and torch are required"):
        load_chemberta()

    load_chemberta.cache_clear()


@pytest.mark.network
def test_compute_from_source_embeds_dataset_smiles(synthetic_dataset) -> None:
    from drevalpy.types.data.feature_source import DrugFeatureSource

    source = DrugFeatureSource(synthetic_dataset, synthetic_dataset.drug_ids)
    featurizer = ChemBertaFeaturizer(max_length=64)

    matrix = featurizer._compute_from_source(source, synthetic_dataset.drug_ids[:2])

    assert matrix.shape[0] == 2
    assert matrix.dtype == np.float32
