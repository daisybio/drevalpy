"""Side-agnostic base for featurizers that emit one dense matrix for one view.

Every dense featurizer in the package - the plain pass-throughs as well as the
ones that reduce, scale or embed - shares the same shape: ask the storage layer
whether the matrix has already been pre-computed, otherwise compute it from the
view, then wrap the result in one numeric block. This module holds that shape
once so subclasses keep only their distinct transform.

Subclass hooks, all optional:

* ``_compute_matrix`` - turn the raw view matrix into the output matrix. The
  default is the identity, which makes a pass-through featurizer a subclass with
  no method bodies at all.
* ``_fit_state`` - learn whatever the transform needs and return the output
  width. The default derives the width from ``_compute_matrix``.
* ``_fit_entity_ids`` - choose the rows to fit on (deduplicated, pair-expanded).
* ``_fetch_hyperparameters`` - the HP setting a stored variant must match.
* ``_block_name`` / ``_block_feature_names`` - name and label the emitted block.

``precompute`` subclasses additionally provide ``_compute_from_source``, used as a
fallback when the declared view is absent from the source altogether.

:class:`DenseViewFeaturizer` is **public**, re-exported from
:mod:`drevalpy.plugin` and covered by that facade's compatibility promise. The
module keeps its leading underscore anyway: ``_discover_modules`` in
``drevalpy/registry/_builtins.py`` skips ``_``-prefixed files, and without it
this base class would be scanned and spuriously registered as a component. So
the *module path* is private and may move; the symbol reached through
``drevalpy.plugin`` may not.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.featurizers._matrix import feature_names_for_view, stack_view_matrix
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.types.data.batch.feature_block import BlockSpec, FeatureBlock, numeric_feature_block
from drevalpy.types.data.feature_source import FeatureSource


class DenseViewFeaturizer(Featurizer):
    """Emit one dense numeric block for one feature view, on either entity side."""

    #: Set on subclasses whose transform is meaningless before ``fit``; they then
    #: raise instead of returning garbage.
    requires_fit: ClassVar[bool] = False
    #: Fit on the deduplicated entity IDs. Right whenever fitting learns a
    #: distribution, where repeated rows would silently reweight it.
    fit_on_unique_ids: ClassVar[bool] = False

    def __init__(self, *, view: str | None = None) -> None:
        """Bind the view this instance reads.

        :param view: View name; ``None`` falls back to the single declared input view.
        """
        self._view = view or self.resolve_input_views()[0]
        self._output_dim = 0
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _fetch_hyperparameters(self) -> dict[str, Any] | None:
        """Return the HP setting a stored variant must match to be reusable.

        ``None`` matches the default (parameter-free) variant, which is right for
        every featurizer whose stored output does not depend on its hyperparameters.

        :returns: HP mapping to match, or ``None``.
        """
        return None

    def _fit_entity_ids(
        self,
        source: FeatureSource,
        entity_ids: np.ndarray | None,
        pair_expanded_ids: np.ndarray | None,
        pair_expanded_es_ids: np.ndarray | None,
    ) -> np.ndarray:
        """Return the entity IDs to fit on.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: Explicit fit IDs, or ``None`` for all identifiers.
        :param pair_expanded_ids: Training entity IDs with duplicates per response pair.
        :param pair_expanded_es_ids: Early-stopping entity IDs with duplicates.
        :returns: IDs whose rows the fit reads.
        """
        _ = pair_expanded_ids, pair_expanded_es_ids
        ids = entity_ids if entity_ids is not None else source.identifiers
        return np.unique(ids) if self.fit_on_unique_ids else ids

    def _fit_state(self, source: FeatureSource, entity_ids: np.ndarray) -> int:
        """Learn the transform state and return the resulting output width.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: IDs chosen by ``_fit_entity_ids``.
        :returns: Output feature dimension after fitting.
        """
        return int(self._compute_matrix(source, self._raw_matrix(source, entity_ids)).shape[1])

    def _compute_matrix(self, source: FeatureSource, matrix: np.ndarray) -> np.ndarray:
        """Turn a raw view *matrix* into this featurizer's output matrix.

        :param source: Feature source the matrix came from.
        :param matrix: Raw view matrix for the requested entity IDs.
        :returns: Output matrix aligned with the same rows.
        """
        _ = source
        return matrix

    def _block_name(self) -> str:
        """Return the name of the single emitted block.

        A declared ``output_block_specs`` wins, so a featurizer can publish under a
        name that differs from the view it reads.

        :returns: Block name.
        """
        specs: tuple[BlockSpec, ...] = getattr(self, "output_block_specs", ())
        return specs[0].name if specs else self._view

    def _block_feature_names(self, source: FeatureSource) -> tuple[str, ...] | None:
        """Return the feature names to attach to the emitted block.

        :param source: Feature source providing views for the entity type.
        :returns: Ordered feature names, or ``None`` when the source has none.
        """
        return feature_names_for_view(source, self._view)

    # ------------------------------------------------------------------
    # Featurizer contract
    # ------------------------------------------------------------------

    def _raw_matrix(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Stack the declared view, falling back to an on-the-fly computation.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: Entity identifiers to stack.
        :returns: Raw view matrix.
        :raises KeyError: If the view is missing and no fallback is available.
        :raises TypeError: If the view cannot be stacked and no fallback is available.
        :raises ValueError: If the view is unusable and no fallback is available.
        """
        try:
            return stack_view_matrix(source, self._view, entity_ids)
        except (KeyError, TypeError, ValueError):
            compute = getattr(self, "_compute_from_source", None)
            if self.precompute and callable(compute):
                return compute(source, entity_ids)
            raise

    def _require_fitted(self) -> None:
        """Reject a transform on a subclass that has state to learn first.

        :raises RuntimeError: If ``requires_fit`` is set and ``fit`` has not run.
        """
        if self.requires_fit and not self._is_fitted:
            msg = f"{type(self).__name__} must be fit before transform"
            raise RuntimeError(msg)

    def _restore_dense_state(self, state: dict[str, object]) -> None:
        """Restore the three fields every dense subclass writes in ``get_state``.

        ``view``, ``output_dim`` and the ``fitted`` flag are set by ``__init__``
        here, not by any subclass, so every subclass ``set_state`` was repeating
        the same three type-guarded reads around its own one fitted object. A key
        the subclass does not write is simply absent and leaves the field alone.

        :param state: Mapping previously returned by ``get_state``.
        """
        view = state.get("view")
        if isinstance(view, str):
            self._view = view
        output_dim = state.get("output_dim")
        if isinstance(output_dim, int):
            self._output_dim = output_dim
        if state.get("fitted"):
            self._is_fitted = True

    def _fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> DenseViewFeaturizer:
        """Record the output width, reusing a pre-computed variant when there is one.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: entity ids.
        :param pair_expanded_ids: Training entity IDs with duplicates per response pair.
        :param pair_expanded_es_ids: Early-stopping entity IDs with duplicates.
        :returns: Result.
        """
        ids = self._fit_entity_ids(source, entity_ids, pair_expanded_ids, pair_expanded_es_ids)
        precomputed = self.fetch_precomputed(source, ids, self._fetch_hyperparameters())
        if precomputed is not None:
            self._output_dim = int(precomputed.shape[1])
            self._on_precomputed_fit(source)
        else:
            self._output_dim = self._fit_state(source, ids)
        self._is_fitted = True
        return self

    def _on_precomputed_fit(self, source: FeatureSource) -> None:
        """Record whatever fit metadata survives a pre-computed short circuit.

        :param source: Feature source carrying the stored variant.
        """
        _ = source

    def _transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Return the dense matrix for *entity_ids*.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: entity ids.
        :returns: Result.
        """
        self._require_fitted()
        precomputed = self.fetch_precomputed(source, entity_ids, self._fetch_hyperparameters())
        if precomputed is not None:
            return precomputed.astype(np.float32)
        return self._compute_matrix(source, self._raw_matrix(source, entity_ids)).astype(np.float32)

    def _transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Wrap the dense matrix in a single named numeric block.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: entity ids.
        :returns: Result.
        """
        return {
            self._block_name(): numeric_feature_block(
                self._transform(source, entity_ids),
                feature_names=self._block_feature_names(source),
            )
        }

    @property
    def output_dim(self) -> int:
        """Return output feature dimension after fitting.

        :returns: Result.
        """
        return self._output_dim
