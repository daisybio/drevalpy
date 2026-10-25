"""Pooled MolGNet drug featurizer for models that need a fixed-length vector per drug.

``molgnetPooled`` reads the node-level MolGNet embeddings that ``molgnet`` serves (a
``(n_atoms, 768)`` matrix per drug, taken from the ``molgnet_features`` view or computed from the
SMILES) and reduces them over the atoms, as RandomForest-AE-MolGNet of drevalpy 1.6.1 does. A tree
ensemble cannot consume a variable-length node matrix. The output block is named
``molgnet_pooled``, which keeps it apart from the ragged ``molgnet_features`` block of DIPK.
"""

from __future__ import annotations

from typing import Any, ClassVar, Final

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers.drug.molgnet import MolGNetDrugFeaturizer
from drevalpy.errors import InvalidInputError
from drevalpy.log import get_logger
from drevalpy.registry.drug_featurizer import register
from drevalpy.types.data.batch.feature_block import BlockSpec, FeaturizerResult, numeric_feature_block
from drevalpy.types.data.feature_source import FeatureSource

_logger = get_logger(__name__)

#: Accepted values of ``pooling``.
POOLINGS: Final = ("mean_max", "mean", "max")


def _unknown_pooling_message(pooling: str) -> str:
    """Build the message for a pooling name that is not supported.

    :param pooling: The rejected name.
    :returns: The error message.
    """
    return f"Unknown MolGNet pooling {pooling!r}; expected one of {', '.join(POOLINGS)}."


def pool_molgnet(nodes: np.ndarray, pooling: str = "mean_max") -> np.ndarray:
    """Aggregate a node-level MolGNet matrix into one fixed-length vector.

    :param nodes: Node embeddings of one molecule, shape ``(n_atoms, n_dim)``.
    :param pooling: ``"mean"``, ``"max"`` or ``"mean_max"`` (the mean followed by the maximum).
    :returns: Vector of ``n_dim`` entries, or ``2 * n_dim`` for ``"mean_max"``.
    :raises InvalidInputError: If *pooling* is not one of :data:`POOLINGS`.
    """
    if pooling == "mean":
        return nodes.mean(axis=0)
    if pooling == "max":
        return nodes.max(axis=0)
    if pooling == "mean_max":
        return np.concatenate((nodes.mean(axis=0), nodes.max(axis=0)))
    raise InvalidInputError(_unknown_pooling_message(pooling))


def molgnet_feature_names(width: int, pooling: str) -> tuple[str, ...]:
    """Name the entries of a pooled MolGNet vector.

    :param width: Length of the pooled vector.
    :param pooling: Pooling the vector came from.
    :returns: ``molgnet_mean_i`` then ``molgnet_max_i`` for ``mean_max``, else ``molgnet_<pooling>_i``.
    """
    if pooling == "mean_max":
        half = width // 2
        return tuple(f"molgnet_mean_{i}" for i in range(half)) + tuple(f"molgnet_max_{i}" for i in range(half))
    return tuple(f"molgnet_{pooling}_{i}" for i in range(width))


@register(
    "molgnetPooled",
    description="MolGNet node embeddings pooled over the atoms (mean, max or both) into one vector per drug.",
    output_format=FeatureFormat.NUMERIC_MATRIX,
)
class MolGNetPooledFeaturizer(MolGNetDrugFeaturizer):
    """Pool the per-atom MolGNet embeddings of each drug into a dense vector."""

    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec("molgnet_pooled", FeatureFormat.NUMERIC_MATRIX),)
    precompute: ClassVar[bool] = False

    def __init__(self, *, view: str = "molgnet_features", pooling: str = "mean_max") -> None:
        """Choose the node view and the pooling.

        :param view: Feature view holding the node matrices; computed from the SMILES when absent.
        :param pooling: ``"mean_max"``, ``"mean"`` or ``"max"``.
        :raises InvalidInputError: If *pooling* is not one of :data:`POOLINGS`.
        """
        if pooling not in POOLINGS:
            raise InvalidInputError(_unknown_pooling_message(pooling))
        super().__init__(view=view)
        self._pooling = pooling

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the tunable pooling.

        :returns: Hyperparameter search space mapping.
        """
        return {"pooling": {"type": "categorical", "choices": list(POOLINGS), "default": "mean_max"}}

    def _nodes(self, source: FeatureSource, drug_key: str) -> np.ndarray | None:
        """Return the node matrix of one drug from the view, or computed from its SMILES.

        :param source: Feature source providing the view and the SMILES.
        :param drug_key: Drug identifier.
        :returns: Node matrix, or ``None`` when neither source has it.
        """
        entity_view = source.get_entity_view(drug_key, self._view)
        if entity_view is not None:
            return np.asarray(entity_view)
        return self._compute_single_embedding(source, drug_key)

    def _pool(self, nodes: np.ndarray) -> np.ndarray:
        """Pool one node matrix to float32.

        :param nodes: Node embeddings, shape ``(n_atoms, n_dim)``.
        :returns: Pooled float32 vector.
        """
        return pool_molgnet(nodes, self._pooling).astype(np.float32)

    def _fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> MolGNetPooledFeaturizer:
        """Pool the node matrices of the fit drugs and record the output width.

        :param source: Feature source providing MolGNet views or SMILES.
        :param entity_ids: Drug identifiers to fit on; all entities when ``None``.
        :param pair_expanded_ids: Unused training IDs with duplicates.
        :param pair_expanded_es_ids: Unused early-stopping IDs.
        :returns: This featurizer.
        """
        _ = pair_expanded_ids, pair_expanded_es_ids
        ids = entity_ids if entity_ids is not None else source.identifiers
        self._features_by_drug = {}
        for drug_id in ids:
            key = str(drug_id)
            if source.get_entity_view(key, self._view) is None and not self._features_by_drug:
                _logger.warning("Computing %s on the fly. Consider precompute().", self.storage_key)
            nodes = self._nodes(source, key)
            if nodes is not None:
                self._features_by_drug[key] = self._pool(nodes)
        if self._features_by_drug:
            self._output_dim = len(next(iter(self._features_by_drug.values())))
        return self

    def _feature_matrix(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Return one pooled vector per drug.

        :param source: Feature source providing MolGNet views or SMILES.
        :param entity_ids: Drug identifiers to transform.
        :returns: Float32 matrix with one row per identifier.
        :raises KeyError: If a drug has no node matrix and none can be computed from its SMILES.
        """
        rows: list[np.ndarray] = []
        for drug_id in entity_ids:
            key = str(drug_id)
            if key not in self._features_by_drug:
                nodes = self._nodes(source, key)
                if nodes is None:
                    msg = f"View {self._view!r} missing for drug {key!r} and on-the-fly computation failed"
                    raise KeyError(msg)
                self._features_by_drug[key] = self._pool(nodes)
            rows.append(self._features_by_drug[key])
        return np.stack(rows) if rows else np.empty((0, self._output_dim), dtype=np.float32)

    def _transform_features(self, source: FeatureSource, entity_ids: np.ndarray) -> FeaturizerResult:
        """Return the pooled vectors as one named ``molgnet_pooled`` block.

        :param source: Feature source providing MolGNet views or SMILES.
        :param entity_ids: Drug identifiers to transform.
        :returns: Result with one numeric block.
        """
        matrix = self._feature_matrix(source, entity_ids)
        block = numeric_feature_block(matrix, feature_names=molgnet_feature_names(matrix.shape[1], self._pooling))
        return FeaturizerResult(entity_ids, {"molgnet_pooled": block}, ("molgnet_pooled",))
