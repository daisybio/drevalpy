"""Molecular graph drug featurizer with on-the-fly fallback."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Self

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.log import get_logger
from drevalpy.registry.featurizer import register
from drevalpy.types.data.batch.feature_block import BlockSpec, FeaturizerResult, graph_feature_block
from drevalpy.types.data.feature_source import FeatureSource, Side

_logger = get_logger(__name__)

_BLOCK = "drug_graph"


@register(
    "drugGraph",
    side=Side.DRUG,
    description="PyG molecular graphs loaded from pre-computed view or computed on the fly via rdkit.",
)
class DrugGraphFeaturizer(Featurizer):
    """Expose drug graphs for graph predictors, with on-the-fly fallback."""

    input_views: ClassVar[tuple[str, ...]] = ("drug_graph",)
    source_views: ClassVar[tuple[str, ...]] = ("canonical_smiles",)

    def __init__(self, *, view: str = "drug_graph", add_hydrogens: bool = False) -> None:
        """Store the graph view name and the on-the-fly graph option.

        :param view: Feature view name containing graph payloads.
        :param add_hydrogens: Whether to add explicit hydrogen atoms to a graph computed from SMILES.
        """
        self._view = view
        self._add_hydrogens = bool(add_hydrogens)

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable hyperparameter specs.

        :returns: HP space mapping.
        """
        return {
            "add_hydrogens": {"type": "categorical", "choices": [True, False], "default": False},
        }

    @classmethod
    def output_blocks(cls, config: Mapping[str, Any]) -> tuple[BlockSpec, ...]:
        """Return the single ``drug_graph`` graph block.

        :param config: Featurizer config; unused.
        :returns: One graph block spec.
        """
        return (BlockSpec(_BLOCK, FeatureFormat.GRAPH),)

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Learn nothing: graphs are read or computed per drug at transform.

        :param source: Drug feature source; unused.
        :param train_ids: Usable training drug IDs; unused.
        :param train_pair_ids: Unused.
        :param es_pair_ids: Unused.
        :returns: This featurizer.
        """
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        """Return one graph per drug: the stored graph when the source has one, else computed from the SMILES.

        :param source: Drug feature source.
        :param ids: Drug IDs, in output order.
        :returns: Result with the ``drug_graph`` block.
        :raises KeyError: If a drug has no stored graph and none can be computed from its SMILES.
        """
        stored = source.graphs(ids) if source.has_view(self._view) else [None] * len(ids)
        smiles = None
        graphs = np.empty(len(ids), dtype=object)
        for row, (drug_id, stored_graph) in enumerate(zip(ids, stored, strict=True)):
            key = str(drug_id)
            graph = as_pyg_graph(stored_graph)
            if graph is None:
                if smiles is None:
                    _logger.info("Computing %s on the fly from SMILES.", _BLOCK)
                    smiles = source.smiles(np.unique(ids))
                graph = self._graph_from_smiles(None if smiles is None else smiles.get(key))
            if graph is None:
                msg = f"View {self._view!r} missing for drug {key!r} and SMILES-based graph computation failed"
                raise KeyError(msg)
            graphs[row] = graph
        return FeaturizerResult(ids, {_BLOCK: graph_feature_block(graphs)}, (_BLOCK,))

    def _graph_from_smiles(self, smiles: object) -> Any:
        """Compute the graph of one SMILES string.

        :param smiles: SMILES string, or a missing value.
        :returns: torch_geometric Data, or ``None`` when *smiles* is missing or unparseable.
        """
        if not smiles or not isinstance(smiles, str):
            return None
        return smiles_to_graph(smiles, add_hydrogens=self._add_hydrogens)


def smiles_to_graph(smiles: str, *, add_hydrogens: bool = False) -> Any:
    """Convert a SMILES string to a torch_geometric Data graph.

    :param smiles: SMILES string.
    :param add_hydrogens: Whether to add explicit hydrogen atoms.
    :returns: torch_geometric.data.Data or None if parsing fails.
    :raises ImportError: If rdkit, torch or torch_geometric is not installed.
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


def _one_hot_encode(value: object, choices: list) -> list[int]:
    """One-hot encode a value given a list of choices, with an extra 'unknown' bin.

    :param value: Value to encode.
    :param choices: Valid choices.
    :returns: One-hot encoded list of length len(choices) + 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def as_pyg_graph(graph: Any) -> Any:
    """Convert a portable h5mu graph mapping to a PyG payload on demand.

    :param graph: A graph mapping, existing PyG payload, or None.
    :returns: The corresponding graph without modifying stored arrays.
    """
    if not isinstance(graph, dict):
        return graph
    import torch
    from torch_geometric.data import Data

    return Data(
        **{
            key: torch.as_tensor(value, dtype=torch.long if key == "edge_index" else torch.float32)
            for key, value in graph.items()
        }
    )
