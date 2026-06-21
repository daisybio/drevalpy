"""XGBoost tabular predictor."""

from __future__ import annotations

from typing import Any

from drevalpy.components.predictors.sklearn_tabular import SklearnTabularPredictor
from drevalpy.components.registry import register_predictor


@register_predictor(
    "xgboost",
    description="XGBoost regressor on concatenated dense features.",
    category="general_purpose",
)
class XGBoostPredictor(SklearnTabularPredictor):
    def _make_estimator(self):
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            msg = "xgboost extra is required for XGBoostPredictor"
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

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 500, "default": 100},
            "max_depth": {"type": "int", "low": 3, "high": 12, "default": 6},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "default": 0.1},
        }
