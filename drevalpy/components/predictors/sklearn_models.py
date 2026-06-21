"""Scikit-learn tabular predictors."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import AdaBoostRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from drevalpy.components.predictors.sklearn_tabular import SklearnTabularPredictor
from drevalpy.components.registry import register_predictor


@register_predictor(
    "elasticNet",
    description="ElasticNet, Ridge, or Lasso on concatenated dense features.",
    category="general_purpose",
)
class ElasticNetPredictor(SklearnTabularPredictor):
    def _make_estimator(self):
        l1_ratio = float(self._h.get("l1_ratio", 0.5))
        alpha = float(self._h.get("alpha", 1.0))
        if l1_ratio == 0.0:
            return Ridge(alpha=alpha)
        if l1_ratio == 1.0:
            return Lasso(alpha=alpha)
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        return {
            "alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True, "default": 1.0},
            "l1_ratio": {"type": "float", "low": 0.0, "high": 1.0, "default": 0.5},
        }


@register_predictor("lasso", description="Lasso regression on dense features.", category="general_purpose")
class LassoPredictor(SklearnTabularPredictor):
    def _make_estimator(self):
        return Lasso(
            alpha=float(self._h.get("alpha", 1.0)),
            max_iter=int(self._h.get("max_iter", 10000)),
            tol=float(self._h.get("tol", 1e-3)),
            selection=str(self._h.get("selection", "random")),
        )


@register_predictor("ridge", description="Ridge regression on dense features.", category="general_purpose")
class RidgePredictor(SklearnTabularPredictor):
    def _make_estimator(self):
        return Ridge(alpha=float(self._h.get("alpha", 1.0)))


@register_predictor(
    "randomForest",
    description="Random forest on concatenated dense features.",
    category="general_purpose",
)
class RandomForestPredictor(SklearnTabularPredictor):
    def _make_estimator(self):
        max_depth_raw = self._h.get("max_depth", 20)
        max_depth = None if max_depth_raw is None else int(max_depth_raw)
        return RandomForestRegressor(
            n_estimators=int(self._h.get("n_estimators", 100)),
            criterion=str(self._h.get("criterion", "squared_error")),
            max_samples=float(self._h.get("max_samples", 0.2)),
            max_depth=max_depth,
            n_jobs=int(self._h.get("n_jobs", -1)),
        )

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        return {
            "n_estimators": {"type": "int", "low": 20, "high": 300, "default": 100},
            "criterion": {
                "type": "categorical",
                "choices": ["squared_error", "absolute_error"],
                "default": "squared_error",
            },
            "max_samples": {"type": "float", "low": 0.1, "high": 0.9, "default": 0.2},
            "max_depth": {"type": "int", "low": 3, "high": 50, "default": 20},
        }


@register_predictor("svr", description="Support vector regression on dense features.", category="general_purpose")
class SVRPredictor(SklearnTabularPredictor):
    def _make_estimator(self):
        return SVR(
            C=float(self._h.get("C", 1.0)),
            epsilon=float(self._h.get("epsilon", 0.1)),
            kernel=str(self._h.get("kernel", "rbf")),
            max_iter=int(self._h.get("max_iter", -1)),
        )


@register_predictor(
    "gradientBoosting",
    description="Histogram gradient boosting on dense features.",
    category="general_purpose",
)
class GradientBoostingPredictor(SklearnTabularPredictor):
    def _make_estimator(self):
        return HistGradientBoostingRegressor(
            max_depth=int(self._h.get("max_depth", 6)),
            learning_rate=float(self._h.get("learning_rate", 0.1)),
            max_iter=int(self._h.get("max_iter", 100)),
        )


@register_predictor(
    "adaboost",
    description="AdaBoost decision tree regressor on dense features.",
    category="general_purpose",
)
class AdaBoostPredictor(SklearnTabularPredictor):
    def _make_estimator(self):
        return AdaBoostRegressor(
            estimator=DecisionTreeRegressor(
                max_depth=int(self._h.get("max_depth", 3)),
                min_samples_split=int(self._h.get("min_samples_split", 2)),
                min_samples_leaf=int(self._h.get("min_samples_leaf", 1)),
            ),
            n_estimators=int(self._h.get("n_estimators", 50)),
            learning_rate=float(self._h.get("learning_rate", 1.0)),
        )


@register_predictor("knn", description="K-nearest neighbors on dense features.", category="general_purpose")
class KNNPredictor(SklearnTabularPredictor):
    def _make_estimator(self):
        return KNeighborsRegressor(
            n_neighbors=int(self._h.get("n_neighbors", 5)),
            weights=str(self._h.get("weights", "distance")),
        )
