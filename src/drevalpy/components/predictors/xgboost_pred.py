"""XGBoost tabular predictor.

``xgboost`` itself is imported inside ``_make_estimator``: ``drevalpy.registry``
imports this module to register the ``xgboost`` predictor on ``import drevalpy``,
and ``xgboost.compat`` pulls in ``sklearn`` (and through it ``scipy.stats``), which
costs ~0.4s. ``_set_xgboost_thread_defaults()`` still runs at *module* scope,
because the environment has to be prepared before anything anywhere imports
``xgboost`` - including a test's own ``importorskip`` - not merely before this
module's own deferred import. Setting four environment variables is free.
See ``tests/test_import_cost_policy.py``.

Everything this shares with ``lightgbm_pred.py`` lives in ``_boosted_trees.py``.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.predictors._boosted_trees import BoostedTreesPredictor
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.registry.predictor import register

_XGBOOST_THREAD_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def _set_xgboost_thread_defaults() -> None:
    """Set conservative native-thread defaults before importing XGBoost.

    XGBoost 3.2 can segfault on macOS when PyTorch/OpenMP has already been
    loaded in the same process; in tests this happened during fit and model
    pickle load. These defaults are applied before importing XGBoost so its
    native runtime initializes single-threaded unless the user has explicitly
    configured thread limits in the environment. See the upstream discussion:
    https://github.com/dmlc/xgboost/issues/11500
    """
    for name, value in _XGBOOST_THREAD_ENV_DEFAULTS.items():
        os.environ.setdefault(name, value)


# Runs at import time, i.e. while `drevalpy.registry` registers builtins, so the
# defaults are in place before any caller reaches `import xgboost`.
_set_xgboost_thread_defaults()


@register(
    "xgboost",
    description="XGBoost regressor on concatenated dense features.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class XGBoostPredictor(BoostedTreesPredictor):
    """XGBoost regressor for dense tabular pair features."""

    # XGBoost tunes a shallower depth ceiling than LightGBM, which is preserved
    # here rather than unified: it changes what a sweep explores.
    boosting_space_overrides: ClassVar[dict[str, dict[str, Any]]] = {
        "max_depth": {"high": 8},
    }

    tuned_hyperparameters: ClassVar[tuple[str, ...]] = ("n_estimators", "max_depth", "learning_rate")

    def _make_estimator(self):
        """Return an unfitted XGBoost regressor.

        :returns: Unfitted ``XGBRegressor`` configured from hyperparameters.
        """
        _set_xgboost_thread_defaults()

        from xgboost import XGBRegressor

        return XGBRegressor(**self._estimator_params(), n_jobs=-1)

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: state.
        :raises PredictorStateError: Raised on invalid input.
        """
        _set_xgboost_thread_defaults()
        super().set_state(state)
        if self._estimator is None:
            msg = "XGBoostPredictor state did not restore a fitted estimator"
            raise PredictorStateError(msg)
