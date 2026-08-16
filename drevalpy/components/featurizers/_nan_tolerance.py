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

from typing import ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.log import get_logger
from drevalpy.types.data.batch.feature_block import FeatureBlock
from drevalpy.types.data.feature_source import FeatureSource

_logger = get_logger(__name__)


class NanToleranceMixin:
    """Decide which entities are usable and keep NaN rows out of the transform."""

    #: Fraction of invalid entities above which a warning is logged.
    nan_threshold: ClassVar[float] = 0.2

    #: Declared by ``FeaturizerDeclarationsMixin``; read here to pick the probe view.
    entity_id_only: ClassVar[bool]
    input_views: ClassVar[tuple[str, ...] | None]

    def _expand_blocks_with_nan(
        self,
        valid_blocks: dict[str, FeatureBlock],
        valid_mask: np.ndarray,
        n_total: int,
    ) -> dict[str, FeatureBlock]:
        """Expand valid-only blocks back to full size, inserting NaN for invalid rows.

        Non-entity-aligned blocks are passed through unchanged.

        :param valid_blocks: Blocks computed on only valid entity IDs.
        :param valid_mask: Boolean mask of shape ``(n_total,)`` (True = valid).
        :param n_total: Total number of entities (valid + invalid).
        :returns: Blocks aligned to the full set of entity IDs.
        """
        expanded: dict[str, FeatureBlock] = {}
        for name, block in valid_blocks.items():
            if not block.entity_aligned:
                expanded[name] = block
                continue
            expanded[name] = _padded_block(block, valid_mask, n_total)
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


def _padded_block(block: FeatureBlock, valid_mask: np.ndarray, n_total: int) -> FeatureBlock:
    """Rebuild one entity-aligned block at full length, padding the invalid rows.

    A numeric matrix is padded with NaN; any other payload is an object array
    padded with ``None``, since there is no NaN for a graph or a token sequence.

    :param block: Block computed on the valid entity IDs only.
    :param valid_mask: Boolean mask of shape ``(n_total,)`` (True = valid).
    :param n_total: Total number of entities (valid + invalid).
    :returns: Block of length *n_total* carrying *block*'s metadata.
    """
    if block.format == FeatureFormat.NUMERIC_MATRIX:
        full = np.full((n_total, block.values.shape[1]), np.nan, dtype=np.float32)
    else:
        full = np.empty(n_total, dtype=object)
        full[:] = None
    full[valid_mask] = block.values
    return FeatureBlock(
        values=full,
        format=block.format,
        feature_names=block.feature_names,
        metadata=block.metadata,
        entity_aligned=True,
    )
