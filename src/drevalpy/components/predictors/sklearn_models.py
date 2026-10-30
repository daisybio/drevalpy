"""Scikit-learn tabular predictors.

The estimator imports live inside each ``_make_estimator`` rather than at module
scope. ``drevalpy.registry`` imports this module to register its nine predictors
on ``import drevalpy``, and importing any part of ``sklearn`` costs ~0.4s because
``sklearn.utils`` pulls in ``scipy.stats``. See ``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from drevalpy.components.predictors.single_drug_sklearn import SingleDrugSklearnPredictor
from drevalpy.components.predictors.sklearn_tabular import SklearnTabularPredictor
from drevalpy.registry.predictor import register


@register(
    "elasticNet",
    description="Elastic Net regression on concatenated dense features.",
)
class ElasticNetPredictor(SklearnTabularPredictor):
    """Elastic net predictor component.

    The defaults are the first grid point of DrEvalPy 1.6.1: ``l1_ratio=0`` (a Ridge fit) with
    ``alpha=1``. At the extremes of ``l1_ratio`` (0 or 1), the estimator falls back to Ridge
    or Lasso; prefer the dedicated ``ridge`` / ``lasso`` predictors when that is
    the intended model. The coordinate-descent settings are scikit-learn's own defaults, as in 1.6.1.
    """

    #: ``l1_ratio`` default of this predictor; the single-drug variant starts from a different mix.
    default_l1_ratio: ClassVar[float] = 0.0

    def _make_estimator(self) -> Any:
        from sklearn.linear_model import ElasticNet, Lasso, Ridge

        l1_ratio = float(self._hyperparameters["l1_ratio"])
        alpha = float(self._hyperparameters["alpha"])
        max_iter = int(self._hyperparameters["max_iter"])
        tol = float(self._hyperparameters["tol"])
        random_state = self._hyperparameters["random_state"]
        if l1_ratio == 0.0:
            # Ridge is not coordinate descent: it takes none of the solver settings.
            return Ridge(alpha=alpha)
        selection = str(self._hyperparameters["selection"])
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
        """Return the hyperparameter space.

        :returns: Search-space specs with a declared default for each parameter; untuned settings are fixed.
        """
        return {
            "alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True, "default": 1.0},
            "l1_ratio": {"type": "float", "low": 0.0, "high": 1.0, "default": cls.default_l1_ratio},
            "max_iter": {"type": "fixed", "default": 1000},
            "tol": {"type": "fixed", "default": 1e-4},
            "selection": {"type": "fixed", "default": "cyclic"},
            "random_state": {"type": "fixed", "default": None},
        }


@register(
    "singleDrugElasticNet",
    description="ElasticNet fitted independently per drug on dense sample features.",
)
class SingleDrugElasticNetPredictor(SingleDrugSklearnPredictor, ElasticNetPredictor):
    """Single-drug ElasticNet predictor component; starts from ``l1_ratio=0.2`` as in 1.6.1."""

    default_l1_ratio: ClassVar[float] = 0.2


@register(
    "lasso",
    description="Lasso regression on dense features.",
)
class LassoPredictor(SklearnTabularPredictor):
    """Lasso predictor component with the solver settings of DrEvalPy 1.6.1.

    ``selection="random"`` is seeded (``random_state=0``): 1.6.1 left it unseeded, so two fits on
    identical data could disagree.
    """

    def _make_estimator(self) -> Any:
        from sklearn.linear_model import Lasso

        return Lasso(
            alpha=float(self._hyperparameters["alpha"]),
            max_iter=int(self._hyperparameters["max_iter"]),
            tol=float(self._hyperparameters["tol"]),
            selection=str(self._hyperparameters["selection"]),
            random_state=self._hyperparameters["random_state"],
        )

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the hyperparameter space.

        :returns: Search-space specs with a declared default for each parameter; untuned settings are fixed.
        """
        return {
            "alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True, "default": 1.0},
            "max_iter": {"type": "fixed", "default": 10000},
            "tol": {"type": "fixed", "default": 1e-3},
            "selection": {"type": "fixed", "default": "random"},
            "random_state": {"type": "fixed", "default": 0},
        }


@register(
    "ridge",
    description="Ridge regression on dense features.",
)
class RidgePredictor(SklearnTabularPredictor):
    """Ridge predictor component."""

    def _make_estimator(self) -> Any:
        from sklearn.linear_model import Ridge

        return Ridge(alpha=float(self._hyperparameters["alpha"]))

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the hyperparameter space.

        :returns: Search-space specs with a declared default for each parameter; untuned settings are fixed.
        """
        return {
            "alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True, "default": 1.0},
        }


@register(
    "randomForest",
    description="Random forest on concatenated dense features.",
)
class RandomForestPredictor(SklearnTabularPredictor):
    """Random forest predictor component."""

    def _make_estimator(self) -> Any:
        from sklearn.ensemble import RandomForestRegressor

        max_depth_raw = self._hyperparameters["max_depth"]
        max_depth = None if max_depth_raw is None else int(max_depth_raw)
        return RandomForestRegressor(
            n_estimators=int(self._hyperparameters["n_estimators"]),
            criterion=str(self._hyperparameters["criterion"]),
            max_samples=float(self._hyperparameters["max_samples"]),
            max_depth=max_depth,
            n_jobs=int(self._hyperparameters["n_jobs"]),
            random_state=self._hyperparameters["random_state"],
        )

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the hyperparameter space.

        :returns: Search-space specs with a declared default for each parameter; untuned settings are fixed.
        """
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 200, "default": 100},
            "max_samples": {"type": "float", "low": 0.1, "high": 0.5, "default": 0.2},
            "max_depth": {"type": "int", "low": 5, "high": 30, "default": 5},
            "criterion": {"type": "fixed", "default": "squared_error"},
            "n_jobs": {"type": "fixed", "default": -1},
            "random_state": {"type": "fixed", "default": 42},
        }


@register(
    "singleDrugRandomForest",
    description="Random forest fitted independently per drug on dense sample features.",
)
class SingleDrugRandomForestPredictor(SingleDrugSklearnPredictor, RandomForestPredictor):
    """Single-drug random-forest predictor component."""


@register(
    "svr",
    description="Support vector regression on dense features.",
)
class SVRPredictor(SklearnTabularPredictor):
    """Support vector regression predictor component."""

    def _make_estimator(self) -> Any:
        from sklearn.svm import SVR

        return SVR(
            C=float(self._hyperparameters["C"]),
            epsilon=float(self._hyperparameters["epsilon"]),
            kernel=str(self._hyperparameters["kernel"]),
            max_iter=int(self._hyperparameters["max_iter"]),
        )

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the hyperparameter space.

        :returns: Search-space specs with a declared default for each parameter; untuned settings are fixed.
        """
        return {
            "C": {"type": "float", "low": 1e-3, "high": 100.0, "log": True, "default": 0.001},
            "epsilon": {"type": "float", "low": 1e-3, "high": 1.0, "log": True, "default": 0.001},
            "kernel": {"type": "categorical", "choices": ["rbf", "linear"], "default": "rbf"},
            "max_iter": {"type": "fixed", "default": 500},
        }


@register(
    "gradientBoosting",
    description="Histogram gradient boosting on dense features.",
)
class GradientBoostingPredictor(SklearnTabularPredictor):
    """Gradient boosting predictor component.

    As in DrEvalPy 1.6.1, the seeded validation split (``validation_fraction``) and
    ``n_iter_no_change`` take effect once scikit-learn turns early stopping on, which it does
    above 10000 training pairs.
    """

    def _make_estimator(self) -> Any:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_depth=int(self._hyperparameters["max_depth"]),
            learning_rate=float(self._hyperparameters["learning_rate"]),
            max_iter=int(self._hyperparameters["max_iter"]),
            validation_fraction=float(self._hyperparameters["validation_fraction"]),
            n_iter_no_change=int(self._hyperparameters["n_iter_no_change"]),
            random_state=int(self._hyperparameters["random_state"]),
        )

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the hyperparameter space.

        :returns: Search-space specs with a declared default for each parameter; untuned settings are fixed.
        """
        return {
            "max_depth": {"type": "int", "low": 3, "high": 30, "default": 5},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True, "default": 0.1},
            "max_iter": {"type": "int", "low": 50, "high": 300, "default": 100},
            "validation_fraction": {"type": "fixed", "default": 0.1},
            "n_iter_no_change": {"type": "fixed", "default": 10},
            "random_state": {"type": "fixed", "default": 42},
        }


@register(
    "adaboost",
    description="AdaBoost decision tree regressor on dense features.",
)
class AdaBoostPredictor(SklearnTabularPredictor):
    """AdaBoost predictor component."""

    def _make_estimator(self) -> Any:
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.tree import DecisionTreeRegressor

        return AdaBoostRegressor(
            estimator=DecisionTreeRegressor(
                max_depth=int(self._hyperparameters["max_depth"]),
                min_samples_split=int(self._hyperparameters["min_samples_split"]),
                min_samples_leaf=int(self._hyperparameters["min_samples_leaf"]),
            ),
            n_estimators=int(self._hyperparameters["n_estimators"]),
            learning_rate=float(self._hyperparameters["learning_rate"]),
        )

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the hyperparameter space.

        :returns: Search-space specs with a declared default for each parameter; untuned settings are fixed.
        """
        return {
            "n_estimators": {"type": "int", "low": 25, "high": 100, "default": 50},
            "max_depth": {"type": "int", "low": 2, "high": 8, "default": 4},
            "min_samples_split": {"type": "int", "low": 2, "high": 6, "default": 2},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 3, "default": 1},
            "learning_rate": {"type": "fixed", "default": 1.0},
        }


@register(
    "knn",
    description="K-nearest neighbors on dense features.",
)
class KNNPredictor(SklearnTabularPredictor):
    """K-nearest neighbors predictor component."""

    def _make_estimator(self) -> Any:
        from sklearn.neighbors import KNeighborsRegressor

        return KNeighborsRegressor(
            n_neighbors=int(self._hyperparameters["n_neighbors"]),
            weights=str(self._hyperparameters["weights"]),
        )

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the hyperparameter space.

        :returns: Search-space specs with a declared default for each parameter; untuned settings are fixed.
        """
        return {
            "n_neighbors": {"type": "int", "low": 3, "high": 15, "default": 5},
            "weights": {"type": "categorical", "choices": ["uniform", "distance"], "default": "distance"},
        }


@register(
    "singleDrugKnn",
    description="K-nearest neighbors per drug on the genes selected by mRMR for that drug.",
)
class SingleDrugKNNPredictor(SingleDrugSklearnPredictor, KNNPredictor):
    """Per-drug kNN on the ``mrmr_k`` features most relevant for that drug's response.

    The distance is computed on the columns the mRMR selection keeps, not on the full
    profile. Standardization, selection and neighbour search are fitted on the training
    pairs of one drug only. A drug with fewer training pairs than ``n_neighbors`` uses as
    many neighbours as it has pairs.
    """

    def _make_estimator(self) -> Any:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        from drevalpy.components.predictors._mrmr import MRMRSelector, NeighborCappedKNN

        mrmr_k = self._hyperparameters["mrmr_k"]
        selector = MRMRSelector(k=int(mrmr_k)) if mrmr_k else "passthrough"
        knn = NeighborCappedKNN(
            n_neighbors=int(self._hyperparameters["n_neighbors"]), weights=str(self._hyperparameters["weights"])
        )
        return Pipeline([("scale", StandardScaler()), ("mrmr", selector), ("knn", knn)])

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the hyperparameter space.

        :returns: Search-space specs with a declared default for each parameter; untuned settings are fixed.
        """
        return {
            "n_neighbors": {"type": "int", "low": 3, "high": 20, "default": 3},
            "weights": {"type": "categorical", "choices": ["uniform", "distance"], "default": "distance"},
            "mrmr_k": {"type": "fixed", "default": 20},
        }
