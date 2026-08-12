"""Throwaway featurizer stand-ins for the ``Dataset`` precompute tests.

``Dataset.precompute`` and ``Dataset._precompute_single`` only read class-level
attributes, a hyperparameter space and ``store``. Deriving from the real
:class:`~drevalpy.components.featurizers.base.Featurizer` ABC would additionally
require a contract and a registry entry, neither of which says anything about the
branches under test, so these stubs implement the duck type instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Defaults describing an eligible cell-line featurizer with no hyperparameters.
_DEFAULT_ATTRS: dict[str, Any] = {
    "storage_key": "stub",
    "side": "cell_line",
    "precompute": True,
    "entity_id_only": False,
    "requires_view": False,
    "input_views": ("gene_expression",),
    "source_views": None,
}


class _StubBase:
    """Shared duck-typed surface every precompute stub needs."""

    def __init__(self, **hyperparameters: Any) -> None:
        """Record the hyperparameters the dataset constructed this variant with."""
        self.hyperparameters = hyperparameters

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, Any]:
        return {}


def _stub_class(name: str, namespace: dict[str, Any], overrides: dict[str, Any]) -> type:
    """Create a fresh stub class so class-level state cannot leak between tests."""
    return type(name, (_StubBase,), {**_DEFAULT_ATTRS, **namespace, **overrides})


def _recording_store(records: list[dict[str, Any]]) -> Any:
    def store(
        self: Any,
        mdata: Any,
        entity_ids: np.ndarray,
        data: np.ndarray,
        hyperparameters: dict[str, Any] | None = None,
    ) -> None:
        records.append({"n_entities": len(entity_ids), "shape": data.shape, "hyperparameters": hyperparameters})

    return store


def stub_featurizer(records: list[dict[str, Any]], **overrides: Any) -> type:
    """Build an independent featurizer class whose ``store`` appends to *records*.

    Args:
        records: List that receives one dict per ``store`` call.
        overrides: Class attributes to replace, e.g. ``side="drug"`` or
            ``precompute=False``. ``compute`` may be set to a callable taking
            ``(source, entity_ids)`` to control what ``_compute_from_source``
            returns or raises.

    Returns:
        A class exposing ``_compute_from_source``, so ``precompute`` takes the
        independent-featurizer shortcut.
    """
    compute = overrides.pop("compute", None)

    def _compute_from_source(self: Any, source: Any, entity_ids: np.ndarray) -> np.ndarray:
        if compute is not None:
            return compute(source, entity_ids)
        return np.zeros((len(entity_ids), 2), dtype=np.float32)

    namespace = {"_compute_from_source": _compute_from_source, "store": _recording_store(records)}
    return _stub_class("StubIndependentFeaturizer", namespace, overrides)


def stub_fit_transform_featurizer(records: list[dict[str, Any]], **overrides: Any) -> type:
    """Build a stub without ``_compute_from_source``, forcing the fit/transform path.

    Args:
        records: List that receives ``{"call": "fit"}`` when ``fit`` runs and one
            dict per ``store`` call.
        overrides: Class attributes to replace, as in :func:`stub_featurizer`.

    Returns:
        A class exposing ``fit``/``transform`` but no ``_compute_from_source``.
    """

    def fit(self: Any, source: Any, *, entity_ids: np.ndarray | None = None) -> Any:
        records.append({"call": "fit"})
        return self

    def transform(self: Any, source: Any, entity_ids: np.ndarray) -> np.ndarray:
        return np.ones((len(entity_ids), 3), dtype=np.float32)

    namespace = {"fit": fit, "transform": transform, "store": _recording_store(records)}
    return _stub_class("StubFitTransformFeaturizer", namespace, overrides)
