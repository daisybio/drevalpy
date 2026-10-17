"""Scikit-learn tabular predictors.

The estimator imports live inside each ``_make_estimator`` rather than at module
scope. ``drevalpy.registry`` imports this module to register its nine predictors
on ``import drevalpy``, and importing any part of ``sklearn`` costs ~0.4s because
``sklearn.utils`` pulls in ``scipy.stats``. See ``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.predictors.single_drug_sklearn import SingleDrugSklearnPredictor
from drevalpy.components.predictors.sklearn_tabular import SklearnTabularPredictor
from drevalpy.registry.predictor import register


@register(
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

    # Coordinate descent here is sensitive to feature scaling, not to the iteration
    # budget: on unstandardized features it fails to converge at any tolerance, and
    # raising max_iter only makes each failure proportionally slower. Keep a modest
    # budget and the looser tolerance/random selection that LassoPredictor uses.
    # ``selection="random"`` needs a seed: unseeded, the coordinate order is drawn
    # from global randomness and two fits on identical data disagree.
    non_tunable_hyperparameters: ClassVar[dict[str, object]] = {
        "max_iter": 2000,
        "tol": 1e-2,
        "selection": "random",
        "random_state": 0,
    }

    def _make_estimator(self):
        from sklearn.linear_model import ElasticNet, Lasso, Ridge

        l1_ratio = float(self._h.get("l1_ratio", 0.5))
        alpha = float(self._h.get("alpha", 1.0))
        max_iter = int(self._h.get("max_iter", 2000))
        tol = float(self._h.get("tol", 1e-2))
        random_state = self._h.get("random_state")
        if l1_ratio == 0.0:
            # Ridge is not coordinate descent and has no ``selection`` parameter.
            return Ridge(alpha=alpha, max_iter=max_iter, tol=tol, random_state=random_state)
        selection = str(self._h.get("selection", "random"))
        if l1_ratio == 1.0:
            return Lasso(
                alpha=alpha,
                max_iter=max_iter,
                tol=tol,
                selection=selection,
                random_state=random_state,
            )
        return ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tol=tol,
            selection=selection,
            random_state=random_state,
        )

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Get hyperparameter space.

        :returns: Result.
        """
        return {
            "alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True, "default": 1.0},
            "l1_ratio": {"type": "float", "low": 0.0, "high": 1.0, "default": 0.5},
        }


@register(
    "singleDrugElasticNet",
    description="ElasticNet fitted independently per drug on dense cell-line features.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class SingleDrugElasticNetPredictor(SingleDrugSklearnPredictor, ElasticNetPredictor):
    """Single-drug ElasticNet predictor component."""


@register(
    "lasso",
    description="Lasso regression on dense features.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class LassoPredictor(SklearnTabularPredictor):
    """Lasso predictor component."""

    non_tunable_hyperparameters: ClassVar[dict[str, object]] = {
        "max_iter": 2000,
        "tol": 1e-2,
        "selection": "random",
        "random_state": 0,
    }

    def _make_estimator(self):
        from sklearn.linear_model import Lasso

        return Lasso(
            alpha=float(self._h.get("alpha", 1.0)),
            max_iter=int(self._h.get("max_iter", 2000)),
            tol=float(self._h.get("tol", 1e-2)),
            selection=str(self._h.get("selection", "random")),
            random_state=self._h.get("random_state"),
        )

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Get hyperparameter space.

        :returns: Result.
        """
        return {
            "alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True, "default": 1.0},
        }


@register(
    "ridge",
    description="Ridge regression on dense features.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class RidgePredictor(SklearnTabularPredictor):
    """Ridge predictor component."""

    def _make_estimator(self):
        from sklearn.linear_model import Ridge

        return Ridge(alpha=float(self._h.get("alpha", 1.0)))

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Get hyperparameter space.

        :returns: Result.
        """
        return {
            "alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True, "default": 1.0},
        }


@register(
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
        from sklearn.ensemble import RandomForestRegressor

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
            "n_estimators": {"type": "int", "low": 50, "high": 200, "default": 100},
            "max_samples": {"type": "float", "low": 0.1, "high": 0.5, "default": 0.2},
            "max_depth": {"type": "int", "low": 5, "high": 25, "default": 15},
        }


@register(
    "singleDrugRandomForest",
    description="Random forest fitted independently per drug on dense cell-line features.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class SingleDrugRandomForestPredictor(SingleDrugSklearnPredictor, RandomForestPredictor):
    """Single-drug random-forest predictor component."""


@register(
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
        from sklearn.svm import SVR

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


@register(
    "gradientBoosting",
    description="Histogram gradient boosting on dense features.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class GradientBoostingPredictor(SklearnTabularPredictor):
    """Gradient boosting predictor component."""

    def _make_estimator(self):
        from sklearn.ensemble import HistGradientBoostingRegressor

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
            "max_depth": {"type": "int", "low": 3, "high": 12, "default": 6},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True, "default": 0.1},
            "max_iter": {"type": "int", "low": 50, "high": 300, "default": 100},
        }


@register(
    "adaboost",
    description="AdaBoost decision tree regressor on dense features.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class AdaBoostPredictor(SklearnTabularPredictor):
    """Ada boost predictor component."""

    # min_samples_split / min_samples_leaf showed no runtime or accuracy signal in
    # benchmark sweeps; they stay overridable but are excluded from tuning so the
    # trial budget goes to max_depth and n_estimators.
    non_tunable_hyperparameters: ClassVar[dict[str, object]] = {
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    }

    def _make_estimator(self):
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.tree import DecisionTreeRegressor

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
            "n_estimators": {"type": "int", "low": 25, "high": 100, "default": 50},
            "max_depth": {"type": "int", "low": 2, "high": 8, "default": 4},
        }


@register(
    "knn",
    description="K-nearest neighbors on dense features.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class KNNPredictor(SklearnTabularPredictor):
    """Knnpredictor component."""

    def _make_estimator(self):
        from sklearn.neighbors import KNeighborsRegressor

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
