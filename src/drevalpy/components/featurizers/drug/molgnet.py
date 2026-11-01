"""MolGNet drug featurizer for DIPK with on-the-fly computation fallback."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Self

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.data.artifacts import get_artifact
from drevalpy.log import get_logger
from drevalpy.registry.featurizer import register
from drevalpy.types.data.batch.feature_block import BlockSpec, FeaturizerResult, ragged_feature_block
from drevalpy.types.data.feature_source import FeatureSource, Side

_logger = get_logger(__name__)

_BLOCK = "molgnet_features"


@register(
    "molgnet",
    description="MolGNet drug embeddings loaded from pre-computed view or computed on the fly.",
)
class MolGNetDrugFeaturizer(Featurizer):
    """Expose variable-size MolGNet tensors with on-the-fly fallback."""

    sides: ClassVar[frozenset[Side]] = frozenset({"drug"})
    input_views: ClassVar[tuple[str, ...]] = ("molgnet_features",)
    source_views: ClassVar[tuple[str, ...]] = ("canonical_smiles",)

    def __init__(self, *, view: str = "molgnet_features") -> None:
        """Store the MolGNet view name.

        :param view: Feature view name containing MolGNet tensors.
        """
        self._view = view

    @classmethod
    def output_blocks(cls, config: Mapping[str, Any]) -> tuple[BlockSpec, ...]:
        """Return the single ragged ``molgnet_features`` block.

        :param config: Featurizer config; unused.
        :returns: One ragged block spec.
        """
        return (BlockSpec(_BLOCK, FeatureFormat.RAGGED_SEQUENCE),)

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Learn nothing: the tensors are read or computed per drug at transform.

        :param source: Drug feature source; unused.
        :param train_ids: Usable training drug IDs; unused.
        :param train_pair_ids: Unused.
        :param es_pair_ids: Unused.
        :returns: This featurizer.
        """
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        """Return one MolGNet node matrix per drug.

        :param source: Drug feature source.
        :param ids: Drug IDs, in output order.
        :returns: Result with the ragged ``molgnet_features`` block.
        """
        rows = molgnet_node_matrices(source, self._view, ids)
        return FeaturizerResult(ids, {_BLOCK: ragged_feature_block(np.array(rows, dtype=object))}, (_BLOCK,))


def molgnet_node_matrices(source: FeatureSource, view: str, ids: np.ndarray) -> list[np.ndarray]:
    """Return the MolGNet node matrix of every drug: from *view* when stored, else computed from its SMILES.

    :param source: Drug feature source.
    :param view: View holding the node matrices.
    :param ids: Drug IDs, in output order.
    :returns: One ``(n_atoms, n_dim)`` matrix per ID.
    :raises KeyError: If a drug has no stored matrix and none can be computed from its SMILES.
    """
    stored = source.view(view, ids) if source.has_view(view) else [None] * len(ids)
    smiles = None
    rows: list[np.ndarray] = []
    for drug_id, entity_view in zip(ids, stored, strict=True):
        key = str(drug_id)
        if entity_view is not None:
            rows.append(np.asarray(entity_view))
            continue
        if smiles is None:
            _logger.info("Computing %s on the fly from SMILES.", _BLOCK)
            smiles = source.smiles(np.unique(ids))
        smi = None if smiles is None else smiles.get(key)
        computed = _compute_molgnet_embedding(smi) if smi and isinstance(smi, str) else None
        if computed is None:
            msg = f"View {view!r} missing for drug {key!r} and on-the-fly computation failed"
            raise KeyError(msg)
        rows.append(computed)
    return rows


def _get_molgnet_checkpoint() -> str:
    """Return the local path to the MolGNet checkpoint, downloading if needed."""
    return str(get_artifact("MolGNet.pt"))


def _compute_molgnet_embedding(smiles: str) -> np.ndarray | None:
    """Compute MolGNet node embedding for a SMILES using the pre-trained checkpoint.

    :param smiles: SMILES string.
    :returns: Numpy array of shape (n_atoms, 768) or None.
    :raises ImportError: If torch or rdkit is not installed.
    """
    try:
        import torch
    except ImportError as err:
        msg = "torch and torch_geometric are required for on-the-fly MolGNet computation"
        raise ImportError(msg) from err
    try:
        from rdkit import Chem
    except ImportError as err:
        msg = "rdkit is required for on-the-fly MolGNet computation: pip install rdkit"
        raise ImportError(msg) from err

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    checkpoint_path = _get_molgnet_checkpoint()

    from drevalpy.components.featurizers.drug._molgnet_network import (
        AddSegId,
        MolGNet,
        SelfLoop,
        mol_to_graph_data_obj_complex,
    )

    graph = mol_to_graph_data_obj_complex(mol)
    self_loop = SelfLoop()
    add_seg = AddSegId()
    prepared = add_seg(self_loop(graph))

    device = torch.device("cpu")
    model = MolGNet(num_layer=5, emb_dim=768, heads=12, num_message_passing=3, drop_ratio=0.0)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    with torch.no_grad():
        emb = model(prepared.to(device))
    return emb.cpu().numpy()
