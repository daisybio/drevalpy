"""Per-drug three-omic helpers shared by MOLIR and SuperFELTR.

Both are single-drug models over the same three sample views, and both need: the record
of which feature names a model was trained on, the column realignment applied at predict
time and the early-stopping index lookup.
The predictor shell that uses them is
:class:`~drevalpy.components.predictors.literature._single_drug_omics.SingleDrugOmicsPredictor`.

Only ``numpy`` is imported at module scope - both predictors are registered on
``import drevalpy``, so the training stack must stay out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from drevalpy.components.predictors.literature._omics_loaders import OmicsSplit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from drevalpy.types.data.batch.model_input_batch import ModelInputBatch

#: The three sample views both models consume, in encoder order.
OMIC_BLOCK_NAMES = ("gene_expression", "mutations", "copy_number_variation_gistic")


@dataclass(frozen=True)
class OmicFeatureNames:
    """The feature names a per-drug model was trained on, one tuple per omic view."""

    gene_expression: tuple[str, ...] | None
    mutations: tuple[str, ...] | None
    copy_number_variation: tuple[str, ...] | None

    def as_tuple(self) -> tuple[tuple[str, ...] | None, ...]:
        """Return the three tuples in :data:`OMIC_BLOCK_NAMES` order.

        :returns: Feature names per view, aligned with ``OMIC_BLOCK_NAMES``.
        """
        return (self.gene_expression, self.mutations, self.copy_number_variation)


@dataclass(frozen=True)
class OmicMatrices:
    """One batch's three entity-level omic matrices, as float32."""

    gene_expression: np.ndarray
    mutations: np.ndarray
    copy_number_variation: np.ndarray

    def widths(self) -> tuple[int, int, int]:
        """Return the feature width of each view.

        :returns: Tuple of ``(expression, mutation, cnv)`` widths.
        """
        return (
            self.gene_expression.shape[1],
            self.mutations.shape[1],
            self.copy_number_variation.shape[1],
        )

    def split(self, pair_idx: np.ndarray, response: np.ndarray) -> OmicsSplit:
        """Pair these entity matrices with a pair-level response and index array.

        :param pair_idx: Row of each matrix to use for each pair.
        :param response: Pair-level response values.
        :returns: A loader-ready split.
        """
        return OmicsSplit(
            gene_expression=self.gene_expression,
            mutations=self.mutations,
            copy_number=self.copy_number_variation,
            response=response,
            pair_idx=pair_idx,
        )


def omic_matrices(batch: ModelInputBatch) -> OmicMatrices:
    """Read the three sample views out of *batch*.

    :param batch: Batch carrying all three blocks.
    :returns: The entity-level matrices as float32.
    """
    values = [np.asarray(batch.sample.blocks[name].values, dtype=np.float32) for name in OMIC_BLOCK_NAMES]
    return OmicMatrices(*values)


def omic_feature_names(batch: ModelInputBatch) -> OmicFeatureNames:
    """Record the feature names of the three sample views of *batch*.

    :param batch: Batch carrying all three blocks.
    :returns: The recorded feature names.
    """
    return OmicFeatureNames(*(batch.sample.blocks[name].feature_names for name in OMIC_BLOCK_NAMES))


def validation_split(matrices: OmicMatrices, batch: ModelInputBatch) -> OmicsSplit | None:
    """Build the early-stopping split over the same entity matrices as training.

    :param matrices: The batch's entity-level omic matrices.
    :param batch: Training batch carrying the early-stopping response.
    :returns: The validation split, or ``None`` when there are fewer than two early-stopping pairs.
    """
    es_resp = batch.early_stopping_response
    if es_resp is None or len(es_resp) < 2:
        return None
    return matrices.split(batch.sample.rows_for(es_resp.sample_ids), np.asarray(es_resp.response, dtype=np.float32))


def aligned_pair_matrices(
    batch: ModelInputBatch,
    feature_names: OmicFeatureNames,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand the three views to pair level and realign them onto the trained order.

    :param batch: Batch to score.
    :param feature_names: Feature names the model was trained on.
    :returns: Tuple of ``(expression, mutation, cnv)`` pair-level matrices.
    """
    pair_idx = batch.sample_pair_idx
    aligned = []
    for name, trained in zip(OMIC_BLOCK_NAMES, feature_names.as_tuple(), strict=True):
        block = batch.sample.blocks[name]
        values = np.asarray(block.values[pair_idx], dtype=np.float32)
        aligned.append(_align(values, trained, block.feature_names))
    return aligned[0], aligned[1], aligned[2]


def _align(
    values: np.ndarray,
    model_features: tuple[str, ...] | None,
    current_features: tuple[str, ...] | None,
) -> np.ndarray:
    """Align omic matrix columns to match the training feature order.

    :param values: Input matrix to realign.
    :param model_features: Feature names expected by the model.
    :param current_features: Feature names in the current batch.
    :returns: Realigned matrix.
    """
    if model_features is None or current_features is None:
        return values
    if len(model_features) == values.shape[1] and model_features == current_features:
        return values
    return _realign_omic_matrix(values, model_features, current_features)


def _realign_omic_matrix(
    values: np.ndarray,
    model_features: Sequence[str] | np.ndarray,
    meta_feature_names: Sequence[str] | np.ndarray,
) -> np.ndarray:
    """Align prediction-time omics columns to the feature order stored on the trained model.

    :param values: Omic feature matrix in incoming column order.
    :param model_features: Feature names used by the trained model.
    :param meta_feature_names: Feature names available in the incoming data.

    :returns: Matrix with columns reordered to match *model_features*; features missing from the incoming data
        are zero.
    """
    if list(meta_feature_names) == list(model_features):
        return values
    realigned = np.zeros((values.shape[0], len(model_features)))
    lookup_table = {feature: i for i, feature in enumerate(meta_feature_names)}
    for column, feature in enumerate(model_features):
        if feature in lookup_table:
            realigned[:, column] = values[:, lookup_table[feature]]
    return realigned
