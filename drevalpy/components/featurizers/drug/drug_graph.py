"""Molecular graph drug featurizer with on-the-fly fallback."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers._feature_source import FeatureSource
from drevalpy.components.featurizers.drug.base import DrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer
from drevalpy.log import get_logger
from drevalpy.types.data.batch.feature_block import BlockSpec, FeatureBlock, graph_feature_block

_logger = get_logger(__name__)


@register_drug_featurizer(
    "drugGraph",
    description="PyG molecular graphs loaded from pre-computed view or computed on the fly via rdkit.",
    contract=FeatureFormat.GRAPH,
)
class DrugGraphFeaturizer(DrugFeaturizer):
    """Expose drug graphs for graph predictors, with on-the-fly fallback."""

    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec("drug_graph", FeatureFormat.GRAPH),)
    input_views: ClassVar[tuple[str, ...]] = ("drug_graph",)
    source_views: ClassVar[tuple[str, ...]] = ("canonical_smiles",)
    precompute: ClassVar[bool] = True

    def __init__(self, *, view: str = "drug_graph", add_hydrogens: bool = False) -> None:
        """Store the graph view name and initialize empty caches.

        :param view: Feature view name containing graph payloads.
        :param add_hydrogens: Whether to add explicit hydrogen atoms to the graph.
        """
        self._view = view
        self._add_hydrogens = bool(add_hydrogens)
        self._graphs: dict[str, object] = {}
        self._output_dim = 0

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable hyperparameter specs.

        :returns: HP space mapping.
        """
        return {
            "add_hydrogens": {"type": "categorical", "choices": [True, False], "default": False},
        }

    def _compute_from_source(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Compute drug graphs from SMILES for all requested entities.

        :param source: Feature source.
        :param entity_ids: Drug identifiers.
        :returns: Object array of graph payloads.
        """
        graphs: list[object] = []
        for drug_id in entity_ids:
            graph = self._compute_graph_from_smiles_for_entity(source, str(drug_id))
            if graph is not None:
                graphs.append(graph)
            else:
                graphs.append(None)
        payloads = np.empty(len(graphs), dtype=object)
        payloads[:] = graphs
        return payloads

    def _fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> DrugGraphFeaturizer:
        """Cache graph payloads and infer node feature width from the first graph.

        :param source: Feature source providing drug graph views.
        :param entity_ids: Drug identifiers to fit on; all entities when ``None``.
        :param pair_expanded_ids: Unused training IDs with duplicates.
        :param pair_expanded_es_ids: Unused early-stopping IDs.
        :returns: Fitted featurizer instance.
        """
        _ = pair_expanded_ids, pair_expanded_es_ids
        ids = entity_ids if entity_ids is not None else source.identifiers
        self._graphs = {}
        has_fallback = False
        for drug_id in ids:
            graph = source.get_entity_view(str(drug_id), self._view)
            if graph is not None:
                self._graphs[str(drug_id)] = graph
            else:
                if not has_fallback:
                    _logger.warning("Computing %s on the fly. Consider ds.precompute().", "drug_graph")
                    has_fallback = True
                computed = self._compute_graph_from_smiles_for_entity(source, str(drug_id))
                if computed is not None:
                    self._graphs[str(drug_id)] = computed
        if self._graphs:
            first = next(iter(self._graphs.values()))
            self._output_dim = int(getattr(first, "num_node_features", 0))
        return self

    def _transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Return one graph payload per drug id.

        :param source: Feature source providing drug graph views.
        :param entity_ids: Drug identifiers to transform.
        :returns: Object array of graph payloads.
        """
        graphs: list[object] = []
        for drug_id in entity_ids:
            drug_key = str(drug_id)
            if drug_key in self._graphs:
                graphs.append(self._graphs[drug_key])
                continue
            graph = source.get_entity_view(drug_key, self._view)
            if graph is not None:
                graphs.append(graph)
            else:
                computed = self._compute_graph_from_smiles_for_entity(source, drug_key)
                if computed is not None:
                    graphs.append(computed)
                else:
                    msg = f"View {self._view!r} missing for drug {drug_key!r} and SMILES-based graph computation failed"
                    raise KeyError(msg)
        payloads = np.empty(len(graphs), dtype=object)
        payloads[:] = graphs
        return payloads

    def _transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Return a single ``drug_graph`` graph block.

        :param source: Feature source providing drug graph views.
        :param entity_ids: Drug identifiers to transform.
        :returns: Mapping with one graph block.
        """
        return {"drug_graph": graph_feature_block(self._transform(source, entity_ids))}

    @property
    def output_dim(self) -> int:
        """Return node feature width inferred during ``fit``.

        :returns: Node feature dimensionality.
        """
        return self._output_dim

    @property
    def graph_by_drug(self) -> dict[str, object]:
        """Return fitted graph payloads keyed by drug id.

        :returns: Cached graph object per drug id.
        """
        return self._graphs

    def _compute_graph_from_smiles_for_entity(self, source: FeatureSource, drug_id: str) -> object | None:
        """Compute a single graph from SMILES for a given drug.

        :param source: Feature source.
        :param drug_id: Drug identifier.
        :returns: torch_geometric Data or None.
        """
        from drevalpy.components.featurizers.drug._smiles_utils import get_smiles_for_entities

        smiles_series = get_smiles_for_entities(source, np.array([drug_id]))
        if smiles_series is None:
            return None
        smi = smiles_series.get(drug_id)
        if not smi or not isinstance(smi, str):
            return None
        return _smiles_to_graph(smi, add_hydrogens=self._add_hydrogens)


def _smiles_to_graph(smiles: str, *, add_hydrogens: bool = False):
    """Convert a SMILES string to a torch_geometric Data graph.

    :param smiles: SMILES string.
    :param add_hydrogens: Whether to add explicit hydrogen atoms.
    :returns: torch_geometric.data.Data or None if parsing fails.
    """
    try:
        from rdkit import Chem
    except ImportError as err:
        msg = "rdkit is required for on-the-fly drug graph computation: pip install rdkit"
        raise ImportError(msg) from err
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as err:
        msg = "torch and torch_geometric are required for on-the-fly drug graph computation"
        raise ImportError(msg) from err

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if add_hydrogens:
        mol = Chem.AddHs(mol)

    atom_feature_defs = {
        "atomic_num": list(range(1, 119)),
        "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
        "num_hs": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "hybridization": [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ],
    }

    bond_feature_defs = {
        "bond_type": [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]
    }

    atom_features_list = []
    for atom in mol.GetAtoms():
        features = []
        features.extend(_one_hot_encode(atom.GetAtomicNum(), atom_feature_defs["atomic_num"]))
        features.extend(_one_hot_encode(atom.GetDegree(), atom_feature_defs["degree"]))
        features.extend(_one_hot_encode(atom.GetFormalCharge(), atom_feature_defs["formal_charge"]))
        features.extend(_one_hot_encode(atom.GetTotalNumHs(), atom_feature_defs["num_hs"]))
        features.extend(_one_hot_encode(atom.GetHybridization(), atom_feature_defs["hybridization"]))
        features.append(atom.GetIsAromatic())
        features.append(atom.IsInRing())
        atom_features_list.append(features)
    x = torch.tensor(atom_features_list, dtype=torch.float)

    edge_indices = []
    edge_features_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        features = []
        features.extend(_one_hot_encode(bond.GetBondType(), bond_feature_defs["bond_type"]))
        features.append(bond.GetIsConjugated())
        features.append(bond.IsInRing())
        edge_indices.extend([[i, j], [j, i]])
        edge_features_list.extend([features, features])

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features_list, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _one_hot_encode(value, choices: list) -> list[int]:
    """One-hot encode a value given a list of choices, with an extra 'unknown' bin.

    :param value: Value to encode.
    :param choices: Valid choices.
    :returns: One-hot encoded list of length len(choices) + 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding
