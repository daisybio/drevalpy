"""XGBoost tabular predictor."""

from __future__ import annotations

import os
from typing import Any

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.predictors.sklearn_tabular import SklearnTabularPredictor
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.components.registry import register_predictor

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


@register_predictor(
    "xgboost",
    description="XGBoost regressor on concatenated dense features.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class XGBoostPredictor(SklearnTabularPredictor):
    """XGBoost regressor for dense tabular pair features."""

    def _make_estimator(self):
        """Return an unfitted XGBoost regressor.

        :returns: Unfitted ``XGBRegressor`` configured from hyperparameters.
        :raises ImportError: If ``xgboost`` is not installed.
        """
        try:
            _set_xgboost_thread_defaults()
            from xgboost import XGBRegressor
        except ImportError as exc:
            msg = "xgboost is required for XGBoostPredictor. Reinstall drevalpy (xgboost is a core dependency)."
            raise ImportError(msg) from exc
        return XGBRegressor(
            n_estimators=int(self._h.get("n_estimators", 100)),
            max_depth=int(self._h.get("max_depth", 6)),
            learning_rate=float(self._h.get("learning_rate", 0.1)),
            subsample=float(self._h.get("subsample", 1.0)),
            colsample_bytree=float(self._h.get("colsample_bytree", 1.0)),
            reg_alpha=float(self._h.get("reg_alpha", 0.0)),
            random_state=int(self._h.get("random_state", 42)),
            n_jobs=-1,
        )

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

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the tunable XGBoost hyperparameter space.

        :returns: Ray Tune-style specs for XGBoost regressor parameters.
        """
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "default": 100},
            "max_depth": {"type": "int", "low": 3, "high": 12, "default": 6},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "default": 0.1},
        }
