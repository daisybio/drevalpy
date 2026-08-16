"""Shared batch factory for the literature predictor tests.

Every literature predictor is exercised against the same shape of input - two
cell lines, two drugs, the four pairs between them, and a checkpoint directory -
so each of those test files had written out the same twelve-keyword
``ModelInputBatch.from_response`` call. What actually differs per predictor is
which feature blocks it consumes, and that is all a caller passes here.

Plain ``_``-prefixed module, per the test-layout rules in ``AGENTS.md``: the
underscore keeps it out of collection, and the mirror policy walks ``drevalpy/``
only, so no mirrored test is demanded for it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.types.data.batch.feature_block import FeatureBlock
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.batch.response_batch import ResponseBatch
from tests.models.synthetic_fixtures import multi_drug_response

CELL_LINE_IDS = np.array(["cl1", "cl2"])
DRUG_IDS = np.array(["d1", "d2"])

#: The four (cell line, drug) pairs of the 2x2 grid, in row-major order.
CELL_LINE_PAIR_IDX = np.array([0, 0, 1, 1])
DRUG_PAIR_IDX = np.array([0, 1, 0, 1])

#: Distinguishes "use :data:`DRUG_PAIR_IDX`" from an explicit ``drug_pair_idx=None``,
#: which is how a single-drug-side predictor says it has no per-drug axis.
_DEFAULT = object()


def two_by_two_batch(
    *,
    cell_line_blocks: dict[str, FeatureBlock],
    drug_blocks: dict[str, FeatureBlock],
    response: ResponseBatch | None = None,
    cell_line_pair_idx: np.ndarray | None = None,
    drug_pair_idx: Any = _DEFAULT,
    early_stopping_response: ResponseBatch | None = None,
    checkpoint_dir: Any = ".",
) -> ModelInputBatch:
    """Build a featurized batch over two cell lines and two drugs.

    :param cell_line_blocks: Cell-line blocks the predictor under test consumes.
    :param drug_blocks: Drug blocks the predictor under test consumes.
    :param response: Training responses; defaults to ``multi_drug_response()``.
    :param cell_line_pair_idx: Per-pair cell-line row index; defaults to the 2x2 grid.
    :param drug_pair_idx: Per-pair drug row index; defaults to the 2x2 grid. Pass
        ``None`` explicitly for a predictor with no per-drug axis.
    :param early_stopping_response: Optional early-stopping responses.
    :param checkpoint_dir: Directory recorded on the ``TrainingContext``.
    :returns: Featurized ``ModelInputBatch``.
    """
    return ModelInputBatch.from_response(
        multi_drug_response() if response is None else response,
        cell_line_entity_ids=CELL_LINE_IDS,
        drug_entity_ids=DRUG_IDS,
        cell_line_features=np.empty((0, 0), dtype=np.float32),
        drug_features=None,
        cell_line_pair_idx=CELL_LINE_PAIR_IDX if cell_line_pair_idx is None else cell_line_pair_idx,
        drug_pair_idx=DRUG_PAIR_IDX if drug_pair_idx is _DEFAULT else drug_pair_idx,
        cell_line_blocks=cell_line_blocks,
        drug_blocks=drug_blocks,
        early_stopping_response=early_stopping_response,
        training_context=TrainingContext(checkpoint_dir=checkpoint_dir),
    )


def early_stopping_response() -> ResponseBatch:
    """Return a two-pair early-stopping response over the same entities.

    :returns: ``ResponseBatch`` for the early-stopping split.
    """
    return ResponseBatch(
        response=np.array([1.5, 2.5]),
        cell_line_ids=CELL_LINE_IDS,
        drug_ids=DRUG_IDS,
    )
