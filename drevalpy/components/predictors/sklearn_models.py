"""Scikit-learn tabular predictors."""

from __future__ import annotations

from typing import Any, ClassVar

from sklearn.ensemble import AdaBoostRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.predictors.single_drug_sklearn import SingleDrugSklearnPredictor
from drevalpy.components.predictors.sklearn_tabular import SklearnTabularPredictor
from drevalpy.components.registry import register_predictor


@register_predictor(
    "elasticNet",
    description="Elastic Net regression on concatenated dense features.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class ElasticNetPredictor(SklearnTabularPredictor):
    """Elastic net predictor component.

    At the extremes of ``l1_ratio`` (0 or 1), the estimator falls back to Ridge
    or Lasso; prefer the dedicated ``ridge`` / ``lasso`` predictors when that is
    the intended model.
    """

    non_tunable_hyperparameters: ClassVar[dict[str, object]] = {
        "max_iter": 1000,
        "tol": 1e-4,
        "selection": "cyclic",
        "random_state": None,
    }

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
        """Get hyperparameter space.

        :returns: Result.
        """
        return {
            "alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True, "default": 1.0},
            "l1_ratio": {"type": "float", "low": 0.0, "high": 1.0, "default": 0.5},
        }


@register_predictor(
    "singleDrugElasticNet",
    description="ElasticNet fitted independently per drug on dense cell-line features.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class SingleDrugElasticNetPredictor(SingleDrugSklearnPredictor, ElasticNetPredictor):
    """Single-drug ElasticNet predictor component."""


@register_predictor(
    "lasso",
    description="Lasso regression on dense features.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class LassoPredictor(SklearnTabularPredictor):
    """Lasso predictor component."""

    non_tunable_hyperparameters: ClassVar[dict[str, object]] = {
        "tol": 1e-3,
        "selection": "random",
    }

    def _make_estimator(self):
        return Lasso(
            alpha=float(self._h.get("alpha", 1.0)),
            max_iter=int(self._h.get("max_iter", 10000)),
            tol=float(self._h.get("tol", 1e-3)),
            selection=str(self._h.get("selection", "random")),
        )

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Get hyperparameter space.

        :returns: Result.
        """
        return {
            "alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True, "default": 1.0},
            "max_iter": {"type": "int", "low": 1000, "high": 20000, "default": 10000},
        }


@register_predictor(
    "ridge",
    description="Ridge regression on dense features.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class RidgePredictor(SklearnTabularPredictor):
    """Ridge predictor component."""

    def _make_estimator(self):
        return Ridge(alpha=float(self._h.get("alpha", 1.0)))

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Get hyperparameter space.

        :returns: Result.
        """
        return {
            "alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True, "default": 1.0},
        }


@register_predictor(
    "randomForest",
    description="Random forest on concatenated dense features.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class RandomForestPredictor(SklearnTabularPredictor):
    """Random forest predictor component."""

    non_tunable_hyperparameters: ClassVar[dict[str, object]] = {
        "n_jobs": -1,
        "random_state": None,
    }

    def _make_estimator(self):
        max_depth_raw = self._h.get("max_depth", 20)
        max_depth = None if max_depth_raw is None else int(max_depth_raw)
        return RandomForestRegressor(
            n_estimators=int(self._h.get("n_estimators", 100)),
            criterion=str(self._h.get("criterion", "squared_error")),
            max_samples=float(self._h.get("max_samples", 0.2)),
            max_depth=max_depth,
            n_jobs=int(self._h.get("n_jobs", -1)),
            random_state=self._h.get("random_state"),
        )

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Get hyperparameter space.

        :returns: Result.
        """
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


@register_predictor(
    "singleDrugRandomForest",
    description="Random forest fitted independently per drug on dense cell-line features.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class SingleDrugRandomForestPredictor(SingleDrugSklearnPredictor, RandomForestPredictor):
    """Single-drug random-forest predictor component."""


@register_predictor(
    "svr",
    description="Support vector regression on dense features.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class SVRPredictor(SklearnTabularPredictor):
    """Svrpredictor component."""

    non_tunable_hyperparameters: ClassVar[dict[str, object]] = {
        "max_iter": -1,
    }

    def _make_estimator(self):
        return SVR(
            C=float(self._h.get("C", 1.0)),
            epsilon=float(self._h.get("epsilon", 0.1)),
            kernel=str(self._h.get("kernel", "rbf")),
            max_iter=int(self._h.get("max_iter", -1)),
        )

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Get hyperparameter space.

        :returns: Result.
        """
        return {
            "C": {"type": "float", "low": 1e-3, "high": 100.0, "log": True, "default": 1.0},
            "epsilon": {"type": "float", "low": 1e-3, "high": 1.0, "log": True, "default": 0.1},
            "kernel": {"type": "categorical", "choices": ["rbf", "linear"], "default": "rbf"},
        }


@register_predictor(
    "gradientBoosting",
    description="Histogram gradient boosting on dense features.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class GradientBoostingPredictor(SklearnTabularPredictor):
    """Gradient boosting predictor component."""

    def _make_estimator(self):
        max_iter = int(self._h.get("max_iter", self._h.get("n_estimators", 100)))
        return HistGradientBoostingRegressor(
            max_depth=int(self._h.get("max_depth", 6)),
            learning_rate=float(self._h.get("learning_rate", 0.1)),
            max_iter=max_iter,
        )

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Get hyperparameter space.

        :returns: Result.
        """
        return {
            "max_depth": {"type": "int", "low": 3, "high": 30, "default": 6},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True, "default": 0.1},
            "max_iter": {"type": "int", "low": 50, "high": 300, "default": 100},
        }


@register_predictor(
    "adaboost",
    description="AdaBoost decision tree regressor on dense features.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class AdaBoostPredictor(SklearnTabularPredictor):
    """Ada boost predictor component."""

    def _make_estimator(self):
        return AdaBoostRegressor(
            estimator=DecisionTreeRegressor(
                max_depth=int(self._h.get("max_depth", 4)),
                min_samples_split=int(self._h.get("min_samples_split", 2)),
                min_samples_leaf=int(self._h.get("min_samples_leaf", 1)),
            ),
            n_estimators=int(self._h.get("n_estimators", 50)),
            learning_rate=float(self._h.get("learning_rate", 1.0)),
        )

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Get hyperparameter space.

        :returns: Result.
        """
        return {
            "n_estimators": {"type": "int", "low": 25, "high": 200, "default": 50},
            "max_depth": {"type": "int", "low": 2, "high": 8, "default": 4},
            "min_samples_split": {"type": "int", "low": 2, "high": 10, "default": 2},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 5, "default": 1},
        }


@register_predictor(
    "knn",
    description="K-nearest neighbors on dense features.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class KNNPredictor(SklearnTabularPredictor):
    """Knnpredictor component."""

    def _make_estimator(self):
        return KNeighborsRegressor(
            n_neighbors=int(self._h.get("n_neighbors", 5)),
            weights=str(self._h.get("weights", "distance")),
        )

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Get hyperparameter space.

        :returns: Result.
        """
        return {
            "n_neighbors": {"type": "int", "low": 3, "high": 15, "default": 5},
            "weights": {"type": "categorical", "choices": ["uniform", "distance"], "default": "distance"},
        }
