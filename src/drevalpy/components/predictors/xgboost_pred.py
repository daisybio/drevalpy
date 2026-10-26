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

from drevalpy.components.contracts.feature_format import FeatureFormat
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
    sample_input_format=FeatureFormat.NUMERIC_MATRIX,
    drug_input_format=FeatureFormat.NUMERIC_MATRIX,
)
class XGBoostPredictor(BoostedTreesPredictor):
    """XGBoost regressor for dense tabular pair features.

    The defaults are the first grid point of DrEvalPy 1.6.1 ``MultiViewXGBoost``: ``max_depth=10``,
    ``subsample=0.8``, ``colsample_bytree=0.6``, ``reg_alpha=0`` and ``reg_lambda=0.1``. Every
    hyperparameter of that grid is tunable here, ``reg_lambda`` included.
    """

    boosting_default_overrides: ClassVar[dict[str, Any]] = {
        "max_depth": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
    }

    boosting_extra_defaults: ClassVar[dict[str, Any]] = {"reg_lambda": 0.1}

    boosting_space_overrides: ClassVar[dict[str, dict[str, Any]]] = {
        "max_depth": {"default": 10},
        "colsample_bytree": {"default": 0.6},
        "reg_lambda": {"default": 0.1},
    }

    tuned_hyperparameters: ClassVar[tuple[str, ...]] = (
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    )

    def _make_estimator(self) -> Any:
        """Return an unfitted XGBoost regressor.

        :returns: Unfitted ``XGBRegressor`` configured from hyperparameters.
        """
        _set_xgboost_thread_defaults()

        from xgboost import XGBRegressor

        return XGBRegressor(**self._estimator_params(), n_jobs=-1)

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: Mapping produced by ``get_state``.
        :raises PredictorStateError: If the restored state contains no fitted estimator.
        """
        _set_xgboost_thread_defaults()
        super().set_state(state)
        if self._estimator is None:
            msg = "XGBoostPredictor state did not restore a fitted estimator"
            raise PredictorStateError(msg)
