"""MolGNet drug featurizer for DIPK with on-the-fly computation fallback."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.core.batch.feature_block import BlockSpec, FeatureBlock, ragged_feature_block
from drevalpy.components.core.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers._feature_source import FeatureSource
from drevalpy.components.featurizers.drug.base import DrugFeaturizer
from drevalpy.components.registry import register_drug_featurizer
from drevalpy.log import get_logger

_logger = get_logger(__name__)


@register_drug_featurizer(
    "molgnet",
    description="MolGNet drug embeddings loaded from pre-computed view or computed on the fly.",
    contract=FeatureFormat.RAGGED_SEQUENCE,
)
class MolGNetDrugFeaturizer(DrugFeaturizer):
    """Expose variable-size MolGNet tensors with on-the-fly fallback."""

    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec("molgnet_features", FeatureFormat.RAGGED_SEQUENCE),
    )
    storage_key: ClassVar[str] = "molgnet_features"
    input_views: ClassVar[tuple[str, ...]] = ("molgnet_features",)
    source_views: ClassVar[tuple[str, ...]] = ("canonical_smiles",)
    precompute: ClassVar[bool] = True

    def __init__(self, *, view: str = "molgnet_features") -> None:
        """Store the MolGNet view name and initialize empty caches.

        :param view: Feature view name containing MolGNet tensors.
        """
        self._view = view
        self._features_by_drug: dict[str, np.ndarray] = {}
        self._output_dim = 0

    def _compute_from_source(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Compute MolGNet embeddings from SMILES for all requested entities.

        :param source: Feature source.
        :param entity_ids: Drug identifiers.
        :returns: Object array of MolGNet embedding tensors.
        """
        rows: list[np.ndarray] = []
        for drug_id in entity_ids:
            computed = self._compute_single_embedding(source, str(drug_id))
            if computed is not None:
                rows.append(computed)
            else:
                rows.append(np.empty((0, 768), dtype=np.float32))
        return np.array(rows, dtype=object)

    def _fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> MolGNetDrugFeaturizer:
        """Cache MolGNet tensors and infer embedding width.

        :param source: Feature source providing MolGNet views.
        :param entity_ids: Drug identifiers to fit on; all entities when ``None``.
        :param pair_expanded_ids: Unused training IDs with duplicates.
        :param pair_expanded_es_ids: Unused early-stopping IDs.
        :returns: Fitted featurizer instance.
        """
        _ = pair_expanded_ids, pair_expanded_es_ids
        ids = entity_ids if entity_ids is not None else source.identifiers
        self._features_by_drug = {}
        has_fallback = False
        for drug_id in ids:
            entity_view = source.get_entity_view(str(drug_id), self._view)
            if entity_view is not None:
                self._features_by_drug[str(drug_id)] = np.asarray(entity_view)
            else:
                if not has_fallback:
                    _logger.warning("Computing %s on the fly. Consider ds.precompute().", self.storage_key)
                    has_fallback = True
                computed = self._compute_single_embedding(source, str(drug_id))
                if computed is not None:
                    self._features_by_drug[str(drug_id)] = computed

        if self._features_by_drug:
            first = next(iter(self._features_by_drug.values()))
            self._output_dim = int(first.shape[1]) if first.ndim == 2 else int(first.size)
        return self

    def _transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Return one MolGNet tensor per drug id.

        :param source: Feature source providing MolGNet views.
        :param entity_ids: Drug identifiers to transform.
        :returns: Object array of MolGNet embedding tensors.
        """
        rows: list[np.ndarray] = []
        for drug_id in entity_ids:
            drug_key = str(drug_id)
            if drug_key in self._features_by_drug:
                rows.append(self._features_by_drug[drug_key])
                continue
            entity_view = source.get_entity_view(drug_key, self._view)
            if entity_view is not None:
                rows.append(np.asarray(entity_view))
            else:
                computed = self._compute_single_embedding(source, drug_key)
                if computed is not None:
                    rows.append(computed)
                else:
                    msg = f"View {self._view!r} missing for drug {drug_key!r} and on-the-fly computation failed"
                    raise KeyError(msg)
        return np.array(rows, dtype=object)

    def _transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Return a single ``molgnet_features`` ragged block.

        :param source: Feature source providing MolGNet views.
        :param entity_ids: Drug identifiers to transform.
        :returns: Mapping with one ragged block.
        """
        return {"molgnet_features": ragged_feature_block(self._transform(source, entity_ids))}

    @property
    def output_dim(self) -> int:
        """Return embedding width inferred during ``fit``.

        :returns: MolGNet embedding dimensionality.
        """
        return self._output_dim

    def _compute_single_embedding(self, source: FeatureSource, drug_id: str) -> np.ndarray | None:
        """Compute MolGNet embedding for a single drug from SMILES.

        Auto-downloads the MolGNet checkpoint on first use.

        :param source: Feature source.
        :param drug_id: Drug identifier.
        :returns: Embedding array or None.
        """
        from drevalpy.components.featurizers.drug._smiles_utils import get_smiles_for_entities

        smiles_series = get_smiles_for_entities(source, np.array([drug_id]))
        if smiles_series is None:
            return None
        smi = smiles_series.get(drug_id)
        if not smi or not isinstance(smi, str):
            return None
        return _compute_molgnet_embedding(smi)


def _get_molgnet_checkpoint() -> str:
    """Return the local path to the MolGNet checkpoint, downloading if needed."""
    from drevalpy.data.artifacts import get_artifact

    return str(get_artifact("MolGNet.pt"))


def _compute_molgnet_embedding(smiles: str) -> np.ndarray | None:
    """Compute MolGNet node embedding for a SMILES using the pre-trained checkpoint.

    :param smiles: SMILES string.
    :returns: Numpy array of shape (n_atoms, 768) or None.
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

    from scripts.featurizer.create_molgnet_embeddings import (
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
