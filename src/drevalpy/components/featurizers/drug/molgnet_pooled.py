"""Pooled MolGNet drug featurizer for models that need a fixed-length vector per drug.

``molgnetPooled`` reads the node-level MolGNet embeddings that ``molgnet`` serves (a
``(n_atoms, 768)`` matrix per drug, taken from the ``molgnet_features`` view or computed from the
SMILES) and reduces them over the atoms, as RandomForest-AE-MolGNet of drevalpy 1.6.1 does. A tree
ensemble cannot consume a variable-length node matrix. The output block is named
``molgnet_pooled``, which keeps it apart from the ragged ``molgnet_features`` block of DIPK.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Final, Self

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers.drug.molgnet import molgnet_node_matrices
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.errors import InvalidInputError
from drevalpy.registry.featurizer import register
from drevalpy.types.data.batch.feature_block import BlockSpec, FeaturizerResult, numeric_feature_block
from drevalpy.types.data.feature_source import FeatureSource, Side

_BLOCK = "molgnet_pooled"

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
    side=Side.DRUG,
    description="MolGNet node embeddings pooled over the atoms (mean, max or both) into one vector per drug.",
    artifact_license="NOASSERTION",
)
class MolGNetPooledFeaturizer(Featurizer):
    """Pool the per-atom MolGNet embeddings of each drug into a dense vector."""

    input_views: ClassVar[tuple[str, ...]] = ("molgnet_features",)
    source_views: ClassVar[tuple[str, ...]] = ("canonical_smiles",)

    def __init__(self, *, view: str = "molgnet_features", pooling: str = "mean_max") -> None:
        """Choose the node view and the pooling.

        :param view: Feature view holding the node matrices; computed from the SMILES when absent.
        :param pooling: ``"mean_max"``, ``"mean"`` or ``"max"``.
        :raises InvalidInputError: If *pooling* is not one of :data:`POOLINGS`.
        """
        if pooling not in POOLINGS:
            raise InvalidInputError(_unknown_pooling_message(pooling))
        self._view = view
        self._pooling = pooling

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the tunable pooling.

        :returns: Hyperparameter search space mapping.
        """
        return {"pooling": {"type": "categorical", "choices": list(POOLINGS), "default": "mean_max"}}

    @classmethod
    def output_blocks(cls, config: Mapping[str, Any]) -> tuple[BlockSpec, ...]:
        """Return the single numeric ``molgnet_pooled`` block.

        :param config: Featurizer config; unused.
        :returns: One numeric block spec.
        """
        return (BlockSpec(_BLOCK, FeatureFormat.NUMERIC_MATRIX),)

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Learn nothing: the node matrices are read or computed and pooled per drug at transform.

        :param source: Drug feature source; unused.
        :param train_ids: Usable training drug IDs; unused.
        :param train_pair_ids: Unused.
        :param es_pair_ids: Unused.
        :returns: This featurizer.
        """
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        """Return the pooled vectors as one named ``molgnet_pooled`` block.

        :param source: Drug feature source.
        :param ids: Drug IDs, in output order.
        :returns: Result with one numeric block.
        """
        rows = [
            pool_molgnet(nodes, self._pooling).astype(np.float32)
            for nodes in molgnet_node_matrices(source, self._view, ids)
        ]
        matrix = np.stack(rows) if rows else np.empty((0, 0), dtype=np.float32)
        block = numeric_feature_block(matrix, feature_names=molgnet_feature_names(matrix.shape[1], self._pooling))
        return FeaturizerResult(ids, {_BLOCK: block}, (_BLOCK,))
