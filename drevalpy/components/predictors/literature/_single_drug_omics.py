"""Per-drug three-omic plumbing shared by MOLIR and SuperFELTR.

Both are single-drug models over the same three cell-line views, and both carried
their own copy of: the per-drug checkpoint directory, the record of which feature
names a model was trained on, the column realignment applied at predict time, the
early-stopping index lookup, and the feature-name (de)serialization.

Only ``numpy`` is imported at module scope - both predictors are registered on
``import drevalpy``, so the training stack must stay out. The leading underscore
keeps the module out of ``registry/_builtins.py::_discover_modules``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from upath import UPath

from drevalpy.components.predictors.literature._omics_loaders import OmicsSplit
from drevalpy.components.predictors.literature.molir._omics import _realign_omic_matrix

if TYPE_CHECKING:
    from collections.abc import Iterator

    from drevalpy.types.data.batch.model_input_batch import ModelInputBatch

#: The three cell-line views both models consume, in encoder order.
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
    """Read the three cell-line views out of *batch*.

    :param batch: Batch carrying all three blocks.
    :returns: The entity-level matrices as float32.
    """
    values = [np.asarray(batch.cell_line_blocks[name].values, dtype=np.float32) for name in OMIC_BLOCK_NAMES]
    return OmicMatrices(*values)


def omic_feature_names(batch: ModelInputBatch) -> OmicFeatureNames:
    """Record the feature names of the three cell-line views of *batch*.

    :param batch: Batch carrying all three blocks.
    :returns: The recorded feature names.
    """
    return OmicFeatureNames(*(batch.cell_line_blocks[name].feature_names for name in OMIC_BLOCK_NAMES))


def checkpoint_dir_for_drug(base_dir: UPath | str, drug_id: str) -> UPath:
    """Return a unique checkpoint directory for a given drug.

    Hashing keeps the segment filesystem-safe: drug ids come from the dataset and
    routinely contain ``/`` and other separators.

    :param base_dir: Base directory for checkpoints.
    :param drug_id: Drug identifier to hash.
    :returns: Path to the drug-specific checkpoint directory.
    """
    digest = hashlib.sha256(drug_id.encode()).hexdigest()[:16]
    return UPath(base_dir) / f"drug_{digest}"


def iter_drug_subsets(batch: ModelInputBatch) -> Iterator[tuple[str, ModelInputBatch]]:
    """Split *batch* per drug, giving each sub-batch its own checkpoint directory.

    :param batch: Full training batch.
    :yields: ``(drug_id, sub_batch)`` for every drug present.
    """
    from drevalpy.components.contracts.training_context import TrainingContext
    from drevalpy.components.predictors.single_drug_routing import iter_drug_masks

    base_dir = batch.training_context.checkpoint_dir
    for drug_id, mask in iter_drug_masks(batch):
        context = TrainingContext(checkpoint_dir=checkpoint_dir_for_drug(base_dir, drug_id))
        yield drug_id, replace(batch.subset_pairs(mask), training_context=context)


def early_stopping_indices(batch: ModelInputBatch) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Map the batch's early-stopping response onto cell-line entity rows.

    :param batch: Training batch.
    :returns: Tuple of ``(val_pair_idx, val_response)``, both ``None`` when there is
        no usable early-stopping split.
    """
    es_resp = batch.early_stopping_response
    if es_resp is None or len(es_resp) < 2:
        return None, None

    entity_map = {str(eid): row for row, eid in enumerate(batch.cell_line_entity_ids)}
    val_idx = np.array([entity_map[str(cl_id)] for cl_id in es_resp.cell_line_ids], dtype=np.intp)
    return val_idx, np.asarray(es_resp.response, dtype=np.float32)


def validation_split(matrices: OmicMatrices, batch: ModelInputBatch) -> OmicsSplit | None:
    """Build the early-stopping split over the same entity matrices as training.

    :param matrices: The batch's entity-level omic matrices.
    :param batch: Training batch carrying the early-stopping response.
    :returns: The validation split, or ``None`` when there is no usable one.
    """
    val_pair_idx, val_response = early_stopping_indices(batch)
    if val_pair_idx is None or val_response is None:
        return None
    return matrices.split(val_pair_idx, val_response)


def aligned_pair_matrices(
    batch: ModelInputBatch,
    feature_names: OmicFeatureNames,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand the three views to pair level and realign them onto the trained order.

    :param batch: Batch to score.
    :param feature_names: Feature names the model was trained on.
    :returns: Tuple of ``(expression, mutation, cnv)`` pair-level matrices.
    """
    pair_idx = batch.cell_line_pair_idx
    aligned = []
    for name, trained in zip(OMIC_BLOCK_NAMES, feature_names.as_tuple(), strict=True):
        block = batch.cell_line_blocks[name]
        values = np.asarray(block.values[pair_idx], dtype=np.float32)
        aligned.append(_align(values, trained, block.feature_names))
    return aligned[0], aligned[1], aligned[2]


def feature_names_payload(feature_names: OmicFeatureNames | None) -> dict[str, list[str] | None]:
    """Render feature names into the serializable form both predictors persist.

    :param feature_names: Recorded feature names, or ``None`` when unknown.
    :returns: Mapping of payload key to feature-name list.
    """
    names = feature_names.as_tuple() if feature_names is not None else (None, None, None)
    keys = ("gene_expression_features", "mutations_features", "copy_number_variation_features")
    return {key: list(value) if value else None for key, value in zip(keys, names, strict=True)}


def feature_names_from_payload(payload: dict[str, Any]) -> OmicFeatureNames:
    """Read feature names back out of a persisted payload.

    :param payload: Deserialized per-drug payload.
    :returns: The recorded feature names.
    """
    keys = ("gene_expression_features", "mutations_features", "copy_number_variation_features")
    values = [tuple(payload[key]) if payload.get(key) else None for key in keys]
    return OmicFeatureNames(*values)


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
