"""LightGBM tabular predictor."""

from __future__ import annotations

from typing import Any

from drevalpy.components.predictors.sklearn_tabular import SklearnTabularPredictor
from drevalpy.components.registry import register_predictor


@register_predictor(
    "lightgbm",
    description="LightGBM regressor on concatenated dense features.",
    category="general_purpose",
)
class LightGBMPredictor(SklearnTabularPredictor):
    """LightGBM regressor for dense tabular pair features."""

    def _make_estimator(self):
        """Return an unfitted LightGBM regressor."""
        try:
            import lightgbm as lgb
        except ImportError as exc:
            msg = "lightgbm extra is required for LightGBMPredictor. " "Install it with: pip install drevalpy[lightgbm]"
            raise ImportError(msg) from exc
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
        """Return the tunable LightGBM hyperparameter space."""
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
