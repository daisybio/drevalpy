"""Test utilities for drevalpy plugins and components.

Everything a plugin's test suite needs to exercise its own components without a
downloaded dataset, beside the in-memory dataset of :func:`drevalpy.synthetic`:

* :func:`build_synthetic_batch` - a featurized batch a predictor can train on.
* :func:`check_plugin` - assert the plugin's entry point loaded and its
  components resolve through the registries.
* The ``check_*`` functions in :mod:`drevalpy.testing.conformance` - assert a
  featurizer or predictor instantiates, fits and transforms, that a fitted one
  round-trips through a pickle, and that a single-drug predictor predicts NaN for
  a drug it was not trained on.

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
    check_featurizer_pickle_round_trip,
    check_predictor_fit_predict,
    check_predictor_instantiates,
    check_predictor_pickle_round_trip,
    check_predictor_single_drug_routing,
    feature_source_for,
)
from .plugins import ENTRY_POINT_GROUP, PluginCheckError, PluginReport, check_plugin

__all__ = [
    "ConformanceError",
    "ENTRY_POINT_GROUP",
    "FEATURIZER_CHECKS",
    "PREDICTOR_CHECKS",
    "PluginCheckError",
    "PluginReport",
    "build_synthetic_batch",
    "check_featurizer_fit_transform",
    "check_featurizer_instantiates",
    "check_featurizer_pickle_round_trip",
    "check_plugin",
    "check_predictor_fit_predict",
    "check_predictor_instantiates",
    "check_predictor_pickle_round_trip",
    "check_predictor_single_drug_routing",
    "feature_source_for",
    "observed_pairs",
]
