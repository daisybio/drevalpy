"""Tests for the SMILESVec drug featurizer; only ``get_artifact`` needs the network."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers.drug.smilesvec import SmilesVecDrugFeaturizer, _smilesvec_embed
from drevalpy.types.data.feature_source import DrugFeatureSource
from tests._import_shims import block_imports
from tests.conftest import MockFeatureSource


class _FakeKeyedVectors:
    def __init__(self, vectors: dict[str, np.ndarray], vector_size: int) -> None:
        self._vectors = vectors
        self.vector_size = vector_size

    @property
    def key_to_index(self) -> dict[str, int]:
        return {word: i for i, word in enumerate(self._vectors)}

    def __getitem__(self, word: str) -> np.ndarray:
        return self._vectors[word]


def test_hyperparameter_space_exposes_the_kmer_length() -> None:
    assert set(SmilesVecDrugFeaturizer.get_hyperparameter_space()) == {"k"}


def test_embed_averages_known_kmers_and_handles_short_and_unknown_smiles() -> None:
    kv = _FakeKeyedVectors({"CCO": np.array([1.0, 3.0]), "CON": np.array([3.0, 5.0]), "CC": np.array([2.0, 4.0])}, 2)

    embedding = _smilesvec_embed("CCOCON", kv, k=3, dim=2)

    np.testing.assert_allclose(embedding, [2.0, 4.0])
    assert embedding.dtype == np.float32
    np.testing.assert_allclose(_smilesvec_embed("CC", kv, k=8, dim=2), [2.0, 4.0])
    np.testing.assert_allclose(_smilesvec_embed("NNNNN", kv, k=3, dim=2), [0.0, 0.0])


def test_compute_from_source_without_smiles_raises() -> None:
    source = MockFeatureSource(features={"d1": {}})

    with pytest.raises(ValueError, match="no SMILES available"):
        SmilesVecDrugFeaturizer()._compute_from_source(source, np.array(["d1"]))


def test_transform_blocks_are_named_smilesvec() -> None:
    source = MockFeatureSource(
        features={"d1": {"smilesvec": np.array([0.1, 0.2])}},
        meta_info={"smilesvec": ["v1", "v2"]},
    )
    ids = np.array(["d1"], dtype=str)
    blocks = SmilesVecDrugFeaturizer().fit(source, entity_ids=ids).transform_blocks(source, ids)

    assert set(blocks) == {"smilesvec"}
    assert blocks["smilesvec"].feature_names == ("v1", "v2")


def test_compute_from_source_reports_missing_gensim(monkeypatch: pytest.MonkeyPatch, synthetic_dataset) -> None:
    block_imports(monkeypatch, "gensim")
    source = DrugFeatureSource(synthetic_dataset, synthetic_dataset.drug_ids)

    with pytest.raises(ImportError, match="gensim is required"):
        SmilesVecDrugFeaturizer()._compute_from_source(source, synthetic_dataset.drug_ids[:1])


@pytest.mark.network
def test_compute_from_source_embeds_dataset_smiles(synthetic_dataset) -> None:
    source = DrugFeatureSource(synthetic_dataset, synthetic_dataset.drug_ids)

    matrix = SmilesVecDrugFeaturizer()._compute_from_source(source, synthetic_dataset.drug_ids[:2])

    assert matrix.shape[0] == 2
    assert matrix.dtype == np.float32
