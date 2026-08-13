"""Test utilities for drevalpy plugins and components.

Everything a plugin's test suite needs to exercise its own components without a
downloaded dataset:

* :func:`build_synthetic_dataset` - an in-memory :class:`~drevalpy.plugin.Dataset`.
* :func:`build_synthetic_batch` - a featurized batch a predictor can train on.
* :func:`check_plugin` - assert the plugin's entry point loaded and its
  components resolve through the registries.
* The ``check_*`` functions in :mod:`drevalpy.testing.conformance` - assert a
  featurizer or predictor instantiates, fits, transforms and round-trips through
  ``get_state``/``set_state``.

This is shipped in the wheel rather than kept in drevalpy's own ``tests/`` tree
precisely so third-party plugins can import it.
"""

from .batch import build_synthetic_batch, observed_pairs
from .conformance import (
    FEATURIZER_CHECKS,
    PREDICTOR_CHECKS,
    ConformanceError,
    check_featurizer_fit_transform,
    check_featurizer_instantiates,
    check_featurizer_state_round_trip,
    check_predictor_fit_predict,
    check_predictor_instantiates,
    check_predictor_state_round_trip,
    feature_source_for,
)
from .plugins import ENTRY_POINT_GROUP, PluginCheckError, PluginReport, check_plugin
from .synthetic import build_synthetic_dataset

__all__ = [
    "ConformanceError",
    "ENTRY_POINT_GROUP",
    "FEATURIZER_CHECKS",
    "PREDICTOR_CHECKS",
    "PluginCheckError",
    "PluginReport",
    "build_synthetic_batch",
    "build_synthetic_dataset",
    "check_featurizer_fit_transform",
    "check_featurizer_instantiates",
    "check_featurizer_state_round_trip",
    "check_plugin",
    "check_predictor_fit_predict",
    "check_predictor_instantiates",
    "check_predictor_state_round_trip",
    "feature_source_for",
    "observed_pairs",
]
