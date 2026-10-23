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

#: At the old default of 1.0 the penalty shrinks every coefficient to zero on
#: standardized features, so the untuned model predicted the training mean.
DEFAULT_ELASTICNET_ALPHA = 0.1


@register(
    "elasticNet",
    description="Elastic Net regression on concatenated dense features.",
    sample_contract=FeatureFormat.NUMERIC_MATRIX,
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

    def _make_estimator(self) -> Any:
        from sklearn.linear_model import ElasticNet, Lasso, Ridge

        l1_ratio = float(self._h["l1_ratio"])
        alpha = float(self._h["alpha"])
        max_iter = int(self._h["max_iter"])
        tol = float(self._h["tol"])
        random_state = self._h["random_state"]
        if l1_ratio == 0.0:
            # Ridge is not coordinate descent and has no ``selection`` parameter.
            return Ridge(alpha=alpha, max_iter=max_iter, tol=tol, random_state=random_state)
        selection = str(self._h["selection"])
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
        """Return the tunable hyperparameter space.

        :returns: Search-space specs with a declared default for each tunable parameter.
        """
        return {
            "alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True, "default": DEFAULT_ELASTICNET_ALPHA},
            "l1_ratio": {"type": "float", "low": 0.0, "high": 1.0, "default": 0.5},
        }


@register(
    "singleDrugElasticNet",
    description="ElasticNet fitted independently per drug on dense sample features.",
    sample_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class SingleDrugElasticNetPredictor(SingleDrugSklearnPredictor, ElasticNetPredictor):
    """Single-drug ElasticNet predictor component."""


@register(
    "lasso",
    description="Lasso regression on dense features.",
    sample_contract=FeatureFormat.NUMERIC_MATRIX,
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

    def _make_estimator(self) -> Any:
        from sklearn.linear_model import Lasso

        return Lasso(
            alpha=float(self._h["alpha"]),
            max_iter=int(self._h["max_iter"]),
            tol=float(self._h["tol"]),
            selection=str(self._h["selection"]),
            random_state=self._h["random_state"],
        )

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the tunable hyperparameter space.

        :returns: Search-space specs with a declared default for each tunable parameter.
        """
        return {
            "alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True, "default": 1.0},
        }


@register(
    "ridge",
    description="Ridge regression on dense features.",
    sample_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class RidgePredictor(SklearnTabularPredictor):
    """Ridge predictor component."""

    def _make_estimator(self) -> Any:
        from sklearn.linear_model import Ridge

        return Ridge(alpha=float(self._h["alpha"]))

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the tunable hyperparameter space.

        :returns: Search-space specs with a declared default for each tunable parameter.
        """
        return {
            "alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True, "default": 1.0},
        }


@register(
    "randomForest",
    description="Random forest on concatenated dense features.",
    sample_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class RandomForestPredictor(SklearnTabularPredictor):
    """Random forest predictor component."""

    non_tunable_hyperparameters: ClassVar[dict[str, object]] = {
        "criterion": "squared_error",
        "n_jobs": -1,
        "random_state": None,
    }

    def _make_estimator(self) -> Any:
        from sklearn.ensemble import RandomForestRegressor

        max_depth_raw = self._h["max_depth"]
        max_depth = None if max_depth_raw is None else int(max_depth_raw)
        return RandomForestRegressor(
            n_estimators=int(self._h["n_estimators"]),
            criterion=str(self._h["criterion"]),
            max_samples=float(self._h["max_samples"]),
            max_depth=max_depth,
            n_jobs=int(self._h["n_jobs"]),
            random_state=self._h["random_state"],
        )

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the tunable hyperparameter space.

        :returns: Search-space specs with a declared default for each tunable parameter.
        """
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 200, "default": 100},
            "max_samples": {"type": "float", "low": 0.1, "high": 0.5, "default": 0.2},
            "max_depth": {"type": "int", "low": 5, "high": 25, "default": 15},
        }


@register(
    "singleDrugRandomForest",
    description="Random forest fitted independently per drug on dense sample features.",
    sample_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class SingleDrugRandomForestPredictor(SingleDrugSklearnPredictor, RandomForestPredictor):
    """Single-drug random-forest predictor component."""


@register(
    "svr",
    description="Support vector regression on dense features.",
    sample_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class SVRPredictor(SklearnTabularPredictor):
    """Support vector regression predictor component."""

    non_tunable_hyperparameters: ClassVar[dict[str, object]] = {
        "max_iter": -1,
    }

    def _make_estimator(self) -> Any:
        from sklearn.svm import SVR

        return SVR(
            C=float(self._h["C"]),
            epsilon=float(self._h["epsilon"]),
            kernel=str(self._h["kernel"]),
            max_iter=int(self._h["max_iter"]),
        )

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the tunable hyperparameter space.

        :returns: Search-space specs with a declared default for each tunable parameter.
        """
        return {
            "C": {"type": "float", "low": 1e-3, "high": 100.0, "log": True, "default": 1.0},
            "epsilon": {"type": "float", "low": 1e-3, "high": 1.0, "log": True, "default": 0.1},
            "kernel": {"type": "categorical", "choices": ["rbf", "linear"], "default": "rbf"},
        }


@register(
    "gradientBoosting",
    description="Histogram gradient boosting on dense features.",
    sample_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class GradientBoostingPredictor(SklearnTabularPredictor):
    """Gradient boosting predictor component."""

    def _make_estimator(self) -> Any:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_depth=int(self._h["max_depth"]),
            learning_rate=float(self._h["learning_rate"]),
            max_iter=int(self._h["max_iter"]),
        )

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the tunable hyperparameter space.

        :returns: Search-space specs with a declared default for each tunable parameter.
        """
        return {
            "max_depth": {"type": "int", "low": 3, "high": 12, "default": 6},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True, "default": 0.1},
            "max_iter": {"type": "int", "low": 50, "high": 300, "default": 100},
        }


@register(
    "adaboost",
    description="AdaBoost decision tree regressor on dense features.",
    sample_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class AdaBoostPredictor(SklearnTabularPredictor):
    """AdaBoost predictor component."""

    # min_samples_split / min_samples_leaf showed no runtime or accuracy signal in
    # benchmark sweeps; they stay overridable but are excluded from tuning so the
    # trial budget goes to max_depth and n_estimators.
    non_tunable_hyperparameters: ClassVar[dict[str, object]] = {
        "learning_rate": 1.0,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    }

    def _make_estimator(self) -> Any:
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.tree import DecisionTreeRegressor

        return AdaBoostRegressor(
            estimator=DecisionTreeRegressor(
                max_depth=int(self._h["max_depth"]),
                min_samples_split=int(self._h["min_samples_split"]),
                min_samples_leaf=int(self._h["min_samples_leaf"]),
            ),
            n_estimators=int(self._h["n_estimators"]),
            learning_rate=float(self._h["learning_rate"]),
        )

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the tunable hyperparameter space.

        :returns: Search-space specs with a declared default for each tunable parameter.
        """
        return {
            "n_estimators": {"type": "int", "low": 25, "high": 100, "default": 50},
            "max_depth": {"type": "int", "low": 2, "high": 8, "default": 4},
        }


@register(
    "knn",
    description="K-nearest neighbors on dense features.",
    sample_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class KNNPredictor(SklearnTabularPredictor):
    """K-nearest neighbors predictor component."""

    def _make_estimator(self) -> Any:
        from sklearn.neighbors import KNeighborsRegressor

        return KNeighborsRegressor(
            n_neighbors=int(self._h["n_neighbors"]),
            weights=str(self._h["weights"]),
        )

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the tunable hyperparameter space.

        :returns: Search-space specs with a declared default for each tunable parameter.
        """
        return {
            "n_neighbors": {"type": "int", "low": 3, "high": 15, "default": 5},
            "weights": {"type": "categorical", "choices": ["uniform", "distance"], "default": "distance"},
        }


@register(
    "singleDrugKnn",
    description="K-nearest neighbors per drug on the genes selected by mRMR for that drug.",
    sample_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class SingleDrugKNNPredictor(SingleDrugSklearnPredictor, KNNPredictor):
    """Per-drug kNN on the ``mrmr_k`` features most relevant for that drug's response.

    The distance is computed on the columns the mRMR selection keeps, not on the full
    profile. Standardization, selection and neighbour search are fitted on the training
    pairs of one drug only. A drug with fewer training pairs than ``n_neighbors`` uses as
    many neighbours as it has pairs.
    """

    non_tunable_hyperparameters: ClassVar[dict[str, object]] = {"mrmr_k": 20}

    def _make_estimator(self) -> Any:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        from drevalpy.components.predictors._mrmr import MRMRSelector, NeighborCappedKNN

        mrmr_k = self._h["mrmr_k"]
        selector = MRMRSelector(k=int(mrmr_k)) if mrmr_k else "passthrough"
        knn = NeighborCappedKNN(n_neighbors=int(self._h["n_neighbors"]), weights=str(self._h["weights"]))
        return Pipeline([("scale", StandardScaler()), ("mrmr", selector), ("knn", knn)])

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the tunable hyperparameter space.

        :returns: Search-space specs with a declared default for each tunable parameter.
        """
        return {
            "n_neighbors": {"type": "int", "low": 3, "high": 20, "default": 3},
            "weights": {"type": "categorical", "choices": ["uniform", "distance"], "default": "distance"},
        }
