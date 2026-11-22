"""Tests for the MolGNet drug featurizers; only the checkpoint load is network-bound."""

from __future__ import annotations

import importlib
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from upath import UPath

import drevalpy.components.featurizers.drug.molgnet as molgnet_module
from drevalpy.components.featurizers.drug.molgnet import MolGNetDrugFeaturizer, _compute_molgnet_embedding
from drevalpy.components.featurizers.drug.molgnet_pooled import (
    MolGNetPooledFeaturizer,
    molgnet_feature_names,
    pool_molgnet,
)
from drevalpy.errors import InvalidInputError
from drevalpy.registry.featurizer import featurizer_registry
from drevalpy.types.data.feature_format import FeatureFormat
from tests._import_shims import block_imports
from tests.components.featurizers.drug._helpers import DrugSource

_D1 = np.arange(6, dtype=np.float32).reshape(2, 3)
_D2 = np.array([[1.0, -2.0, 4.0], [3.0, 0.0, -1.0], [-5.0, 8.0, 2.0]], dtype=np.float32)
_IDS = np.array(["d1", "d2"], dtype=str)


def _ragged_source() -> DrugSource:
    return DrugSource({"molgnet_features": {"d1": _D1, "d2": _D2}})


def _record_embeddings(monkeypatch: pytest.MonkeyPatch, embedding: np.ndarray | None = _D1) -> list[str]:
    calls: list[str] = []

    def compute(smiles: str) -> np.ndarray | None:
        calls.append(smiles)
        return embedding

    monkeypatch.setattr(molgnet_module, "_compute_molgnet_embedding", compute)
    return calls


def test_both_molgnet_featurizers_are_registered_drug_featurizers() -> None:
    assert featurizer_registry("drug").get("molgnet") is MolGNetDrugFeaturizer
    assert featurizer_registry("drug").get("molgnetPooled") is MolGNetPooledFeaturizer
    assert MolGNetDrugFeaturizer.side == "drug"
    assert MolGNetPooledFeaturizer.side == "drug"


def test_molgnet_returns_one_ragged_tensor_per_drug() -> None:
    source = _ragged_source()

    result = MolGNetDrugFeaturizer().fit(source, _IDS).transform(source, _IDS)

    assert set(result.blocks) == {"molgnet_features"}
    block = result.blocks["molgnet_features"]
    assert block.format is FeatureFormat.RAGGED_SEQUENCE
    assert block.values.shape == (2,)
    np.testing.assert_allclose(block.values[0], _D1)
    np.testing.assert_allclose(block.values[1], _D2)
    assert MolGNetDrugFeaturizer.output_blocks({})[0].format is FeatureFormat.RAGGED_SEQUENCE


def test_molgnet_reads_drugs_it_was_not_fitted_on() -> None:
    source = _ragged_source()
    featurizer = MolGNetDrugFeaturizer().fit(source, _IDS[:1])

    values = featurizer.transform(source, _IDS[1:]).blocks["molgnet_features"].values

    # A single tensor makes ``np.array(rows, dtype=object)`` a (1, n, dim) array, so re-cast the row.
    np.testing.assert_allclose(np.asarray(values[0], dtype=np.float32), _D2)


def test_molgnet_reads_the_configured_view() -> None:
    source = DrugSource({"custom": {"d1": _D2}})

    values = MolGNetDrugFeaturizer(view="custom").fit(source, _IDS[:1]).transform(source, _IDS[:1])

    np.testing.assert_allclose(np.asarray(values.blocks["molgnet_features"].values[0], dtype=np.float32), _D2)


def test_molgnet_without_view_or_smiles_raises_a_key_error() -> None:
    source = DrugSource({"other": {"d1": [1.0]}})
    featurizer = MolGNetDrugFeaturizer().fit(source, _IDS[:1])

    with pytest.raises(KeyError, match="on-the-fly computation failed"):
        featurizer.transform(source, _IDS[:1])


def test_molgnet_does_not_compute_a_missing_smiles(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record_embeddings(monkeypatch)
    source = DrugSource(smiles={"d1": float("nan")})

    with pytest.raises(KeyError, match="'d1'"):
        MolGNetDrugFeaturizer().transform(source, _IDS[:1])
    assert calls == []


def test_molgnet_compute_embedding_rejects_an_unparseable_smiles() -> None:
    assert _compute_molgnet_embedding("not-a-smiles") is None


@pytest.mark.parametrize(
    ("blocked", "message"), [("torch", "torch and torch_geometric are required"), ("rdkit", "rdkit is required")]
)
def test_molgnet_compute_embedding_names_the_missing_dependency(
    monkeypatch: pytest.MonkeyPatch, blocked: str, message: str
) -> None:
    block_imports(monkeypatch, blocked)

    with pytest.raises(ImportError, match=message):
        _compute_molgnet_embedding("CCO")


def test_the_molgnet_network_module_names_the_missing_rdkit_on_import(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "drevalpy.components.featurizers.drug._molgnet_network"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    block_imports(monkeypatch, "rdkit")

    with pytest.raises(ImportError, match="Please install rdkit package for MolGNet featurizer") as caught:
        importlib.import_module(module_name)

    assert isinstance(caught.value.__cause__, ImportError)


def test_molgnet_computes_drugs_without_a_stored_view_from_their_smiles(
    monkeypatch: pytest.MonkeyPatch, smiles_only: DrugSource
) -> None:
    calls = _record_embeddings(monkeypatch)
    drug_ids = smiles_only._ids[:2]
    featurizer = MolGNetDrugFeaturizer().fit(smiles_only, drug_ids)
    assert calls == []

    values = featurizer.transform(smiles_only, drug_ids).blocks["molgnet_features"].values

    smiles = smiles_only.smiles(drug_ids)
    assert smiles is not None
    assert calls == list(smiles)
    np.testing.assert_allclose(np.asarray(values[0], dtype=np.float32), _D1)


def test_molgnet_computes_only_the_drugs_missing_from_the_view(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record_embeddings(monkeypatch)
    source = DrugSource({"molgnet_features": {"d2": _D2}}, smiles={"d1": "CCO", "d2": "CCN"})

    values = MolGNetDrugFeaturizer().transform(source, _IDS).blocks["molgnet_features"].values

    assert calls == ["CCO"]
    np.testing.assert_allclose(values[0], _D1)
    np.testing.assert_allclose(values[1], _D2)


def test_molgnet_featurizers_pickle() -> None:
    source = _ragged_source()
    for featurizer in (MolGNetDrugFeaturizer(), MolGNetPooledFeaturizer(pooling="max")):
        fitted = featurizer.fit(source, _IDS)
        restored = pickle.loads(pickle.dumps(fitted))  # noqa: S301 -- pickled on this line
        np.testing.assert_array_equal(
            restored.transform(source, _IDS).blocks[fitted.output_blocks({})[0].name].values[1],
            fitted.transform(source, _IDS).blocks[fitted.output_blocks({})[0].name].values[1],
        )


def test_the_checkpoint_is_the_molgnet_artifact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    requested: list[str] = []

    def get_artifact(name: str) -> UPath:
        requested.append(name)
        return UPath(tmp_path) / name

    monkeypatch.setattr(molgnet_module, "get_artifact", get_artifact)

    assert molgnet_module._get_molgnet_checkpoint() == str(UPath(tmp_path) / "MolGNet.pt")
    assert requested == ["MolGNet.pt"]


@pytest.fixture(scope="module")
def random_molgnet(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], dict[str, UPath]]:
    """A randomly initialised MolGNet, as the state dict it was built from and as saved checkpoints.

    :returns: The state dict and, per checkpoint format, the file the featurizer's loader is pointed at.
    """
    import torch

    from drevalpy.components.featurizers.drug._molgnet_network import MolGNet

    torch.manual_seed(0)
    state_dict = MolGNet(num_layer=5, emb_dim=768, heads=12, num_message_passing=3, drop_ratio=0.0).state_dict()
    directory = UPath(tmp_path_factory.mktemp("molgnet"))
    paths = {"plain": directory / "plain.pt", "wrapped": directory / "wrapped.pt"}
    torch.save(state_dict, str(paths["plain"]))
    torch.save({"state_dict": state_dict, "epoch": 3}, str(paths["wrapped"]))
    return state_dict, paths


@pytest.mark.slow
class TestComputeEmbeddingFromCheckpoint:
    """``_compute_molgnet_embedding`` against a random checkpoint, so no download is needed."""

    @staticmethod
    def _expected(state_dict: dict[str, Any], smiles: str) -> np.ndarray:
        import torch
        from rdkit import Chem

        from drevalpy.components.featurizers.drug._molgnet_network import MolGNet, mol_to_graph_data_obj_complex

        model = MolGNet(num_layer=5, emb_dim=768, heads=12, num_message_passing=3, drop_ratio=0.0)
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            return model(mol_to_graph_data_obj_complex(Chem.MolFromSmiles(smiles))).numpy()

    @pytest.mark.parametrize("checkpoint_format", ["plain", "wrapped"])
    def test_either_checkpoint_format_gives_one_768_wide_row_per_atom(
        self, monkeypatch: pytest.MonkeyPatch, random_molgnet, checkpoint_format: str
    ) -> None:
        state_dict, paths = random_molgnet
        monkeypatch.setattr(molgnet_module, "_get_molgnet_checkpoint", lambda: str(paths[checkpoint_format]))

        embedding = _compute_molgnet_embedding("CCO")

        assert embedding is not None
        assert embedding.shape == (3, 768)
        assert np.isfinite(embedding).all()
        np.testing.assert_allclose(embedding, self._expected(state_dict, "CCO"), rtol=1e-4, atol=1e-5)

    def test_an_invalid_smiles_returns_none_without_reading_the_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unexpected() -> str:
            raise AssertionError("the checkpoint must not be fetched for an invalid SMILES")

        monkeypatch.setattr(molgnet_module, "_get_molgnet_checkpoint", unexpected)

        assert _compute_molgnet_embedding("not-a-smiles") is None

    def test_the_featurizer_computes_a_missing_view_with_it(
        self, monkeypatch: pytest.MonkeyPatch, random_molgnet
    ) -> None:
        state_dict, paths = random_molgnet
        monkeypatch.setattr(molgnet_module, "_get_molgnet_checkpoint", lambda: str(paths["plain"]))
        source = DrugSource(smiles={"d1": "CCO"})

        values = MolGNetDrugFeaturizer().transform(source, _IDS[:1]).blocks["molgnet_features"].values

        np.testing.assert_allclose(
            np.asarray(values[0], dtype=np.float32), self._expected(state_dict, "CCO"), rtol=1e-4, atol=1e-5
        )


@pytest.mark.network
def test_molgnet_computes_an_embedding_from_the_checkpoint() -> None:
    from drevalpy.components.featurizers.drug.molgnet import _get_molgnet_checkpoint

    assert _get_molgnet_checkpoint().endswith("MolGNet.pt")
    embedding = _compute_molgnet_embedding("CCO")
    assert embedding is not None
    assert embedding.shape == (3, 768)


@pytest.mark.parametrize(
    ("pooling", "expected"),
    [
        ("mean_max", np.concatenate([_D2.mean(axis=0), _D2.max(axis=0)])),
        ("mean", _D2.mean(axis=0)),
        ("max", _D2.max(axis=0)),
    ],
)
def test_pool_molgnet_reductions(pooling: str, expected: np.ndarray) -> None:
    np.testing.assert_array_equal(pool_molgnet(_D2, pooling), expected)


def test_unknown_pooling_is_rejected_by_the_function_and_the_constructor() -> None:
    with pytest.raises(InvalidInputError, match="Unknown MolGNet pooling 'sum'"):
        pool_molgnet(_D2, "sum")
    with pytest.raises(InvalidInputError, match="mean_max, mean, max"):
        MolGNetPooledFeaturizer(pooling="median")


def test_molgnet_feature_names_follow_the_pooling() -> None:
    assert molgnet_feature_names(4, "mean_max") == (
        "molgnet_mean_0",
        "molgnet_mean_1",
        "molgnet_max_0",
        "molgnet_max_1",
    )
    assert molgnet_feature_names(2, "max") == ("molgnet_max_0", "molgnet_max_1")


@pytest.mark.parametrize(("pooling", "width"), [("mean_max", 6), ("mean", 3), ("max", 3)])
def test_pooled_transform_pools_every_drug(pooling: str, width: int) -> None:
    source = _ragged_source()
    featurizer = MolGNetPooledFeaturizer(pooling=pooling).fit(source, _IDS)

    matrix = featurizer.transform(source, _IDS).matrix

    assert matrix.shape == (2, width)
    assert matrix.dtype == np.float32
    np.testing.assert_array_equal(matrix[1], pool_molgnet(_D2, pooling))
    assert featurizer.transform(source, np.array([], dtype=str)).matrix.shape[0] == 0


def test_pooled_transform_emits_one_named_numeric_block() -> None:
    source = _ragged_source()

    blocks = MolGNetPooledFeaturizer().fit(source, _IDS).transform(source, _IDS).blocks

    assert set(blocks) == {"molgnet_pooled"}
    assert blocks["molgnet_pooled"].format is FeatureFormat.NUMERIC_MATRIX
    assert blocks["molgnet_pooled"].feature_names == molgnet_feature_names(6, "mean_max")
    assert MolGNetPooledFeaturizer.output_blocks({})[0].name == "molgnet_pooled"


def test_equal_atom_counts_still_pool_per_drug_and_unfitted_drugs_are_read() -> None:
    source = DrugSource({"molgnet_features": {"a": _D1, "b": _D1 + 10, "c": _D2}})
    featurizer = MolGNetPooledFeaturizer(pooling="mean").fit(source, np.array(["a", "b"]))

    matrix = featurizer.transform(source, np.array(["a", "b", "c"])).matrix

    np.testing.assert_array_equal(matrix[:2], np.stack([_D1.mean(axis=0), (_D1 + 10).mean(axis=0)]))
    np.testing.assert_array_equal(matrix[2], _D2.mean(axis=0))


def test_pooled_computes_a_missing_view_from_the_smiles(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record_embeddings(monkeypatch)
    source = DrugSource({"molgnet_features": {"d2": _D2}}, smiles={"d1": "CCO", "d2": "CCN"})
    featurizer = MolGNetPooledFeaturizer(pooling="mean").fit(source, _IDS)
    assert calls == []

    np.testing.assert_array_equal(featurizer.transform(source, np.array(["d1"])).matrix[0], _D1.mean(axis=0))
    assert calls == ["CCO"]


def test_pooled_unavailable_drug_raises_a_key_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _record_embeddings(monkeypatch, embedding=None)
    unavailable = DrugSource({"molgnet_features": {"d1": _D1}}, smiles={"d1": "CCO", "d2": "CCN"})
    fitted = MolGNetPooledFeaturizer().fit(unavailable, np.array(["d1"]))
    with pytest.raises(KeyError, match="d2"):
        fitted.transform(unavailable, np.array(["d2"]))


def test_hyperparameter_space_lists_the_poolings() -> None:
    assert MolGNetPooledFeaturizer.get_hyperparameter_space()["pooling"]["choices"] == ["mean_max", "mean", "max"]
    assert MolGNetPooledFeaturizer.default_hyperparameters() == {"pooling": "mean_max"}
