"""Tests for the MolGNet ragged drug featurizer.

Mirrors :mod:`drevalpy.components.featurizers.drug.molgnet`. Everything except
``_compute_molgnet_embedding``'s checkpoint load is exercised offline: the class
is a dict cache over ``FeatureSource.get_entity_view``, so a mock source serving
ragged arrays covers ``_fit`` / ``_transform`` / ``_transform_blocks`` without
touching the 300 MB ``MolGNet.pt`` artifact.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.drug.molgnet import (
    MolGNetDrugFeaturizer,
    _compute_molgnet_embedding,
)
from tests._import_shims import block_imports
from tests.conftest import MockFeatureSource

_D1 = np.arange(6, dtype=np.float32).reshape(2, 3)
_D2 = np.arange(9, dtype=np.float32).reshape(3, 3)


def _ragged_source() -> MockFeatureSource:
    """Source serving two MolGNet tensors of differing atom counts."""
    return MockFeatureSource(
        features={
            "d1": {"molgnet_features": _D1},
            "d2": {"molgnet_features": _D2},
        }
    )


def test_molgnet_fit_infers_embedding_width_from_the_first_tensor() -> None:
    featurizer = MolGNetDrugFeaturizer()

    featurizer.fit(_ragged_source(), entity_ids=np.array(["d1", "d2"], dtype=str))

    assert featurizer.output_dim == 3


def test_molgnet_output_dim_is_zero_before_fit() -> None:
    assert MolGNetDrugFeaturizer().output_dim == 0


def test_molgnet_transform_returns_one_tensor_per_drug() -> None:
    source = _ragged_source()
    ids = np.array(["d1", "d2"], dtype=str)
    featurizer = MolGNetDrugFeaturizer().fit(source, entity_ids=ids)

    payloads = featurizer.transform(source, ids)

    assert payloads.shape == (2,)
    np.testing.assert_allclose(payloads[0], _D1)
    np.testing.assert_allclose(payloads[1], _D2)


def test_molgnet_transform_reads_uncached_drugs_from_the_source() -> None:
    source = _ragged_source()
    featurizer = MolGNetDrugFeaturizer().fit(source, entity_ids=np.array(["d1"], dtype=str))

    payloads = featurizer.transform(source, np.array(["d2"], dtype=str))

    # A single tensor makes ``np.array(rows, dtype=object)`` build a (1, n, dim)
    # object array rather than a 1-element array of tensors, so re-cast the row.
    np.testing.assert_allclose(np.asarray(payloads[0], dtype=np.float32), _D2)


def test_molgnet_transform_blocks_emits_a_single_ragged_block() -> None:
    source = _ragged_source()
    ids = np.array(["d1", "d2"], dtype=str)
    featurizer = MolGNetDrugFeaturizer().fit(source, entity_ids=ids)

    blocks = featurizer.transform_blocks(source, ids)

    assert set(blocks) == {"molgnet_features"}
    assert blocks["molgnet_features"].format is FeatureFormat.RAGGED_SEQUENCE
    assert blocks["molgnet_features"].values.shape == (2,)


def test_molgnet_honours_a_custom_view_name() -> None:
    source = MockFeatureSource(features={"d1": {"custom": _D1}})
    featurizer = MolGNetDrugFeaturizer(view="custom")

    featurizer.fit(source, entity_ids=np.array(["d1"], dtype=str))

    assert featurizer.output_dim == 3


def test_molgnet_fit_flattens_one_dimensional_tensors() -> None:
    source = MockFeatureSource(features={"d1": {"molgnet_features": np.arange(4, dtype=np.float32)}})
    featurizer = MolGNetDrugFeaturizer()

    featurizer.fit(source, entity_ids=np.array(["d1"], dtype=str))

    assert featurizer.output_dim == 4


def test_molgnet_fit_over_all_identifiers_when_none_are_given() -> None:
    featurizer = MolGNetDrugFeaturizer().fit(_ragged_source())

    assert featurizer.output_dim == 3


def test_molgnet_fit_skips_drugs_it_can_neither_read_nor_compute() -> None:
    source = MockFeatureSource(features={"d1": {}})

    featurizer = MolGNetDrugFeaturizer().fit(source, entity_ids=np.array(["d1"], dtype=str))

    assert featurizer.output_dim == 0


def test_molgnet_transform_raises_for_a_drug_it_cannot_resolve() -> None:
    source = MockFeatureSource(features={"d1": {}})
    featurizer = MolGNetDrugFeaturizer().fit(source, entity_ids=np.array(["d1"], dtype=str))

    with pytest.raises(KeyError, match="on-the-fly computation failed"):
        featurizer.transform(source, np.array(["d1"], dtype=str))


def test_molgnet_compute_from_source_emits_empty_rows_without_smiles() -> None:
    source = MockFeatureSource(features={"d1": {}, "d2": {}})
    featurizer = MolGNetDrugFeaturizer()

    computed = featurizer._compute_from_source(source, np.array(["d1", "d2"], dtype=str))

    assert computed.shape[0] == 2
    assert computed[0].shape == (0, 768)


def test_molgnet_compute_embedding_returns_none_for_unparseable_smiles() -> None:
    assert _compute_molgnet_embedding("not-a-smiles") is None


@pytest.mark.parametrize(
    ("blocked", "message"),
    [
        pytest.param("torch", "torch and torch_geometric are required", id="torch"),
        pytest.param("rdkit", "rdkit is required", id="rdkit"),
    ],
)
def test_molgnet_compute_embedding_names_the_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
    blocked: str,
    message: str,
) -> None:
    block_imports(monkeypatch, blocked)

    with pytest.raises(ImportError, match=message):
        _compute_molgnet_embedding("CCO")


def test_molgnet_compute_single_embedding_returns_none_without_smiles() -> None:
    source = MockFeatureSource(features={"d1": {}})

    assert MolGNetDrugFeaturizer()._compute_single_embedding(source, "d1") is None


def _patch_embedding(monkeypatch: pytest.MonkeyPatch, rows: int = 4) -> np.ndarray:
    """Replace the checkpoint-backed embedding with a fixed tensor.

    ``_compute_molgnet_embedding`` is the only artifact-download boundary in the
    on-the-fly path; stubbing it exercises the surrounding fallback branches
    offline.
    """
    import drevalpy.components.featurizers.drug.molgnet as molgnet_module

    embedding = np.ones((rows, 3), dtype=np.float32)
    monkeypatch.setattr(molgnet_module, "_compute_molgnet_embedding", lambda smiles: embedding)
    return embedding


def test_molgnet_fit_computes_missing_drugs_on_the_fly(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_dataset,
) -> None:
    from drevalpy.types.data.feature_source import DrugFeatureSource

    class _NoStoredTensorSource(DrugFeatureSource):
        def get_entity_view(self, entity_id: str, view: str) -> None:
            return None

    embedding = _patch_embedding(monkeypatch)
    source = _NoStoredTensorSource(synthetic_dataset, synthetic_dataset.drug_ids)
    drug_ids = synthetic_dataset.drug_ids[:2]

    featurizer = MolGNetDrugFeaturizer().fit(source, entity_ids=drug_ids)

    assert featurizer.output_dim == embedding.shape[1]
    assert set(featurizer._features_by_drug) == set(drug_ids)


def test_molgnet_transform_computes_missing_drugs_on_the_fly(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_dataset,
) -> None:
    from drevalpy.types.data.feature_source import DrugFeatureSource

    class _NoStoredTensorSource(DrugFeatureSource):
        def get_entity_view(self, entity_id: str, view: str) -> None:
            return None

    embedding = _patch_embedding(monkeypatch)
    source = _NoStoredTensorSource(synthetic_dataset, synthetic_dataset.drug_ids)

    payloads = MolGNetDrugFeaturizer().transform(source, synthetic_dataset.drug_ids[:1])

    np.testing.assert_allclose(np.asarray(payloads[0], dtype=np.float32), embedding)


def test_molgnet_compute_from_source_uses_the_computed_embedding(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_dataset,
) -> None:
    from drevalpy.types.data.feature_source import DrugFeatureSource

    embedding = _patch_embedding(monkeypatch)
    source = DrugFeatureSource(synthetic_dataset, synthetic_dataset.drug_ids)

    computed = MolGNetDrugFeaturizer()._compute_from_source(source, synthetic_dataset.drug_ids[:2])

    assert computed.shape[0] == 2
    np.testing.assert_allclose(np.asarray(computed[0], dtype=np.float32), embedding)


def test_molgnet_compute_single_embedding_returns_none_for_a_non_string_smiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    import drevalpy.components.featurizers.drug.molgnet as molgnet_module

    monkeypatch.setattr(
        molgnet_module,
        "get_smiles_for_entities",
        lambda source, entity_ids: pd.Series({"d1": float("nan")}),
    )

    assert MolGNetDrugFeaturizer()._compute_single_embedding(MockFeatureSource(features={}), "d1") is None


@pytest.mark.network
def test_molgnet_checkpoint_path_resolves() -> None:
    from drevalpy.components.featurizers.drug.molgnet import _get_molgnet_checkpoint

    assert _get_molgnet_checkpoint().endswith("MolGNet.pt")


@pytest.mark.network
def test_molgnet_computes_an_embedding_from_the_checkpoint() -> None:
    embedding = _compute_molgnet_embedding("CCO")

    assert embedding is not None
    assert embedding.shape == (3, 768)
