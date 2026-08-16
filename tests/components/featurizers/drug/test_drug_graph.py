"""Tests for graph drug featurizer payload handling and on-the-fly fallback."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from drevalpy.components.featurizers.drug.drug_graph import (
    DrugGraphFeaturizer,
    _one_hot_encode,
    _smiles_to_graph,
)
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import DrugFeatureSource
from tests._import_shims import block_imports
from tests.conftest import MockFeatureSource


def test_drug_graph_featurizer_preserves_graph_payloads() -> None:
    graph = Data(
        x=torch.ones((2, 3)),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
    )
    features = MockFeatureSource({"d1": {"drug_graph": graph}})
    featurizer = DrugGraphFeaturizer().fit(features, entity_ids=np.array(["d1"]))

    block = featurizer.transform_blocks(features, np.array(["d1"]))["drug_graph"]

    assert block.values.shape == (1,)
    assert block.values[0] is graph


def test_drug_graph_infers_node_feature_width_from_the_first_graph() -> None:
    graph = Data(x=torch.ones((2, 3)), edge_index=torch.tensor([[0], [1]], dtype=torch.long))
    features = MockFeatureSource({"d1": {"drug_graph": graph}})

    featurizer = DrugGraphFeaturizer().fit(features, entity_ids=np.array(["d1"]))

    assert featurizer.output_dim == 3
    assert set(featurizer.graph_by_drug) == {"d1"}


def test_drug_graph_hyperparameter_space_exposes_add_hydrogens() -> None:
    assert set(DrugGraphFeaturizer.get_hyperparameter_space()) == {"add_hydrogens"}


def test_drug_graph_transform_reads_uncached_drugs_from_the_source() -> None:
    graph = Data(x=torch.ones((2, 3)), edge_index=torch.tensor([[0], [1]], dtype=torch.long))
    other = Data(x=torch.zeros((1, 3)), edge_index=torch.empty((2, 0), dtype=torch.long))
    features = MockFeatureSource({"d1": {"drug_graph": graph}, "d2": {"drug_graph": other}})
    featurizer = DrugGraphFeaturizer().fit(features, entity_ids=np.array(["d1"]))

    payloads = featurizer.transform(features, np.array(["d2"]))

    assert payloads[0] is other


def test_drug_graph_fit_skips_drugs_it_can_neither_read_nor_compute() -> None:
    features = MockFeatureSource({"d1": {}})

    featurizer = DrugGraphFeaturizer().fit(features, entity_ids=np.array(["d1"]))

    assert featurizer.graph_by_drug == {}
    assert featurizer.output_dim == 0


def test_drug_graph_transform_raises_for_a_drug_it_cannot_resolve() -> None:
    features = MockFeatureSource({"d1": {}})
    featurizer = DrugGraphFeaturizer().fit(features, entity_ids=np.array(["d1"]))

    with pytest.raises(KeyError, match="graph computation failed"):
        featurizer.transform(features, np.array(["d1"]))


class _NoStoredGraphSource(DrugFeatureSource):
    """Dataset-backed drug source that reports no stored graph payloads.

    ``DrugFeatureSource.get_entity_view`` raises ``KeyError`` rather than
    returning ``None`` for an absent view, so this override is the only way to
    reach the SMILES-based on-the-fly fallback in ``_fit`` / ``_transform``.
    """

    def get_entity_view(self, entity_id: str, view: str) -> None:
        """Report every graph view as absent."""
        return None


def test_drug_graph_fit_computes_graphs_on_the_fly_for_missing_views(synthetic_dataset: Dataset) -> None:
    source = _NoStoredGraphSource(synthetic_dataset, synthetic_dataset.drug_ids)
    drug_ids = synthetic_dataset.drug_ids[:2]

    featurizer = DrugGraphFeaturizer().fit(source, entity_ids=drug_ids)

    assert set(featurizer.graph_by_drug) == set(drug_ids)
    assert featurizer.output_dim > 0


def test_drug_graph_transform_computes_graphs_on_the_fly_for_missing_views(
    synthetic_dataset: Dataset,
) -> None:
    source = _NoStoredGraphSource(synthetic_dataset, synthetic_dataset.drug_ids)

    payloads = DrugGraphFeaturizer().transform(source, synthetic_dataset.drug_ids[:1])

    assert isinstance(payloads[0], Data)


def test_drug_graph_compute_from_source_emits_none_without_smiles() -> None:
    features = MockFeatureSource({"d1": {}, "d2": {}})

    payloads = DrugGraphFeaturizer()._compute_from_source(features, np.array(["d1", "d2"]))

    assert payloads.tolist() == [None, None]


def test_drug_graph_computes_graphs_from_dataset_smiles(synthetic_dataset: Dataset) -> None:
    source = DrugFeatureSource(synthetic_dataset, synthetic_dataset.drug_ids)
    drug_ids = synthetic_dataset.drug_ids[:2]

    payloads = DrugGraphFeaturizer()._compute_from_source(source, drug_ids)

    assert payloads.shape == (2,)
    assert all(isinstance(payload, Data) for payload in payloads)


def test_smiles_to_graph_returns_none_for_unparseable_smiles() -> None:
    assert _smiles_to_graph("this is not a molecule") is None


def test_smiles_to_graph_emits_empty_edges_for_a_single_atom() -> None:
    graph = _smiles_to_graph("C")

    assert graph is not None
    assert graph.x.shape[0] == 1
    assert graph.edge_index.shape == (2, 0)
    assert graph.edge_attr.shape == (0, 6)


def test_smiles_to_graph_adds_explicit_hydrogens_on_request() -> None:
    without = _smiles_to_graph("C", add_hydrogens=False)
    with_hs = _smiles_to_graph("C", add_hydrogens=True)

    assert without is not None
    assert with_hs is not None
    assert with_hs.x.shape[0] == without.x.shape[0] + 4
    assert with_hs.edge_index.shape[1] == 8


def test_one_hot_encode_uses_the_trailing_bin_for_unknown_values() -> None:
    assert _one_hot_encode("z", ["a", "b"]) == [0, 0, 1]


def test_one_hot_encode_sets_the_matching_position() -> None:
    assert _one_hot_encode("b", ["a", "b"]) == [0, 1, 0]


@pytest.mark.parametrize(
    ("blocked", "message"),
    [
        pytest.param(("rdkit",), "rdkit is required", id="rdkit"),
        pytest.param(("torch",), "torch and torch_geometric are required", id="torch-and-geometric"),
    ],
)
def test_smiles_to_graph_names_the_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
    blocked: tuple[str, ...],
    message: str,
) -> None:
    """``"torch"`` as a prefix blocks ``torch_geometric`` as well."""
    block_imports(monkeypatch, *blocked)

    with pytest.raises(ImportError, match=message):
        _smiles_to_graph("CCO")


def test_drug_graph_compute_from_source_returns_none_for_a_non_string_smiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    import drevalpy.components.featurizers.drug.drug_graph as drug_graph_module

    monkeypatch.setattr(
        drug_graph_module,
        "get_smiles_for_entities",
        lambda source, entity_ids: pd.Series({"d1": float("nan")}),
    )
    features = MockFeatureSource({"d1": {}})

    assert DrugGraphFeaturizer()._compute_graph_from_smiles_for_entity(features, "d1") is None
