"""LightGBM tabular predictor."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.predictors.sklearn_tabular import SklearnTabularPredictor
from drevalpy.registry.predictor import register


@register(
    "lightgbm",
    description="LightGBM regressor on concatenated dense features.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class LightGBMPredictor(SklearnTabularPredictor):
    """LightGBM regressor for dense tabular pair features."""

    def _make_estimator(self):
        """Return an unfitted LightGBM regressor.

        :returns: Unfitted ``LGBMRegressor`` configured from hyperparameters.
        """
        return lgb.LGBMRegressor(
            n_estimators=int(self._h.get("n_estimators", 100)),
            learning_rate=float(self._h.get("learning_rate", 0.1)),
            max_depth=int(self._h.get("max_depth", 6)),
            num_leaves=int(self._h.get("num_leaves", 63)),
            subsample=float(self._h.get("subsample", 0.8)),
            colsample_bytree=float(self._h.get("colsample_bytree", 0.8)),
            reg_alpha=float(self._h.get("reg_alpha", 0.0)),
            reg_lambda=float(self._h.get("reg_lambda", 0.0)),
            random_state=int(self._h.get("random_state", 42)),
            n_jobs=int(self._h.get("n_jobs", -1)),
            verbosity=-1,
        )

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the tunable LightGBM hyperparameter space.

        :returns: Ray Tune-style specs for LightGBM regressor parameters.
        """
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "default": 100},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True, "default": 0.1},
            "max_depth": {"type": "int", "low": 3, "high": 12, "default": 6},
            "num_leaves": {"type": "int", "low": 15, "high": 255, "default": 63},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0, "default": 0.8},
            "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0, "default": 0.8},
            "reg_alpha": {"type": "float", "low": 0.0, "high": 10.0, "default": 0.0},
            "reg_lambda": {"type": "float", "low": 0.0, "high": 10.0, "default": 0.0},
        }
