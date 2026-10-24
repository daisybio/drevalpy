"""NaN tolerance for featurizers: detect all-NaN entities, warn, re-insert.

Every public ``fit``/``transform`` on :class:`~drevalpy.plugin.Featurizer` brackets
its subclass hook with the same three steps - work out which entities have usable
rows, complain when too few do, and pad the result back to full length. That
policy is independent of what any featurizer computes, and it reaches nothing on
the class beyond the three declarations below, which is why it lives apart from
the fit/transform contract in ``base.py``.

The module is ``_``-prefixed so ``_discover_modules`` in
``drevalpy/registry/_builtins.py`` does not scan it as a component.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import ClassVar

import numpy as np

from drevalpy.log import get_logger
from drevalpy.types.data.batch.feature_block import FeatureBlock
from drevalpy.types.data.feature_source import FeatureSource

_logger = get_logger(__name__)


class NanToleranceMixin:
    """Decide which entities are usable and keep NaN rows out of the transform."""

    #: Fraction of invalid entities above which a warning is logged. A fifth is a heuristic
    #: for "a lot of the cohort is unusable"; subclasses may override it.
    nan_threshold: ClassVar[float] = 0.2

    #: Declared by ``FeaturizerDeclarationsMixin``; read here to pick the probe view.
    entity_id_only: ClassVar[bool]
    input_views: ClassVar[tuple[str, ...] | None]

    def _expand_blocks_with_nan(
        self,
        valid_blocks: Mapping[str, FeatureBlock],
        valid_mask: np.ndarray,
    ) -> dict[str, FeatureBlock]:
        """Expand valid-only blocks back to full size, inserting NaN for invalid rows.

        Non-entity-aligned blocks are passed through unchanged.

        :param valid_blocks: Blocks computed on only valid entity IDs.
        :param valid_mask: Boolean mask of shape ``(n_total,)`` (True = valid).
        :returns: Blocks aligned to the full set of entity IDs.
        """
        expanded: dict[str, FeatureBlock] = {}
        for name, block in valid_blocks.items():
            if not block.entity_aligned:
                expanded[name] = block
                continue
            expanded[name] = _padded_block(block, valid_mask)
        return expanded

    def _detect_valid(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Return a boolean mask indicating which entities have non-NaN features.

        Default: entity_id_only featurizers treat all as valid; view-based
        featurizers check the first input view for all-NaN rows.

        :param source: Feature source.
        :param entity_ids: Entity IDs to check.
        :returns: Boolean array of shape ``(len(entity_ids),)``.
        """
        all_valid = np.ones(len(entity_ids), dtype=bool)
        if self.entity_id_only:
            return all_valid

        view = getattr(self, "_view", None)
        if view is None and self.input_views:
            view = self.input_views[0]
        if view is None:
            return all_valid

        try:
            matrix = source.get_view_matrix(view, entity_ids)
        except (KeyError, TypeError, ValueError):
            return all_valid

        if matrix.ndim != 2 or matrix.dtype.kind not in ("f", "i", "u"):
            return all_valid

        return ~np.all(np.isnan(matrix), axis=1)

    def _warn_if_above_threshold(self, valid_mask: np.ndarray, context: str) -> None:
        """Log a warning when the fraction of invalid entities exceeds the threshold.

        :param valid_mask: Boolean array (True = valid).
        :param context: Human-readable label for the warning message.
        """
        if len(valid_mask) == 0:
            return
        invalid_frac = 1.0 - valid_mask.mean()
        if invalid_frac > self.nan_threshold:
            _logger.warning(
                "%s: %.0f%% of inputs are invalid (threshold: %.0f%%)",
                context,
                invalid_frac * 100,
                self.nan_threshold * 100,
            )


def _padded_block(block: FeatureBlock, valid_mask: np.ndarray) -> FeatureBlock:
    """Restore entity rows with dtype-appropriate missing values.

    :param block: Valid-only entity payload.
    :param valid_mask: Requested positions containing usable rows.
    :returns: Padded block retaining feature names and metadata.
    """
    numeric = block.values.dtype.kind in "biufc"
    full = np.full(
        (len(valid_mask), *block.values.shape[1:]),
        np.nan if numeric else None,
        dtype=np.result_type(block.values.dtype, np.float32) if numeric else object,
    )
    full[valid_mask] = block.values
    return replace(block, values=full)
