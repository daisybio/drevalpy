"""Materialize featurizer blocks into FeatureDataset views."""

from __future__ import annotations

import numpy as np

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.feature_block import FeatureBlock
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.literature._feature_dataset_from_batch import feature_dataset_from_blocks
from drevalpy.datasets.dataset import FeatureDataset


def materialize_block_inputs(
    predictor: object,
    batch: ModelInputBatch,
    *,
    required_cell_line_blocks: tuple[str, ...],
    required_drug_blocks: tuple[str, ...],
    requires_drug_featurizer: bool,
    validate_drug_graphs: bool = False,
) -> tuple[FeatureDataset, FeatureDataset | None]:
    """Build cell-line and optional drug FeatureDatasets from batch blocks.

    :param predictor: Predictor requesting the materialized views.
    :param batch: Structured input batch with entity ids and feature blocks.
    :param required_cell_line_blocks: Cell-line block names that must be present.
    :param required_drug_blocks: Drug block names that must be present when drugs are required.
    :param requires_drug_featurizer: Whether drug features should be materialized.
    :param validate_drug_graphs: When ``True``, validate graph blocks before returning.

    :returns: Cell-line dataset and optional drug dataset.
    """
    cell_lines = _dataset_from_blocks(
        predictor,
        batch.cell_line_entity_ids,
        batch.cell_line_blocks,
        required=required_cell_line_blocks,
        side="cell_line",
    )
    if not requires_drug_featurizer:
        return cell_lines, None
    drugs = _dataset_from_blocks(
        predictor,
        batch.drug_entity_ids,
        batch.drug_blocks,
        required=required_drug_blocks,
        side="drug",
    )
    if validate_drug_graphs:
        _validate_drug_graphs(predictor, batch.drug_blocks)
    return cell_lines, drugs


def _dataset_from_blocks(
    predictor: object,
    entity_ids: np.ndarray | None,
    blocks: dict[str, FeatureBlock],
    *,
    required: tuple[str, ...],
    side: str,
) -> FeatureDataset:
    if entity_ids is None:
        msg = f"{predictor.__class__.__name__} requires {side} entity ids"
        raise RuntimeError(msg)
    missing = [name for name in required if name not in blocks]
    if missing:
        msg = f"{predictor.__class__.__name__} missing {side} block(s) {missing}; required={list(required)}"
        raise ValueError(msg)
    selected = {name: blocks[name] for name in required}
    return feature_dataset_from_blocks(entity_ids, selected)


def _validate_drug_graphs(predictor: object, blocks: dict[str, FeatureBlock]) -> None:
    """Reject malformed graph blocks before a graph predictor sees them.

    :param predictor: Predictor used only for error-message context.
    :param blocks: Drug-side feature blocks from the batch.

    :raises ValueError: If ``drug_graph`` is present but malformed.
    """
    block = blocks.get("drug_graph")
    if block is None:
        return
    if block.format != FeatureFormat.GRAPH:
        raise ValueError(
            f"{predictor.__class__.__name__} requires drug_graph blocks with format "
            f"{FeatureFormat.GRAPH.value!r}, got {block.format.value!r}"
        )
    for graph in block.values:
        if not hasattr(graph, "x") or not hasattr(graph, "edge_index"):
            raise ValueError(
                f"{predictor.__class__.__name__} received an invalid drug graph; " "each graph needs x and edge_index"
            )
