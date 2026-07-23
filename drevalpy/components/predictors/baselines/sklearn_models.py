"""Concrete multi-drug sklearn baseline adapters."""

from drevalpy.components.predictors.baselines.sklearn_base import (
    SingleDrugSklearnModel,
    SklearnModel,
)


class ElasticNetModel(SklearnModel):
    """ElasticNet model for drug response prediction."""

    @classmethod
    def get_model_name(cls) -> str:
        """Return the public model name."""
        return "ElasticNet"


class RandomForest(SklearnModel):
    """Random forest model for drug response prediction."""

    @classmethod
    def get_model_name(cls) -> str:
        """Return the public model name."""
        return "RandomForest"


class SVMRegressor(SklearnModel):
    """Support vector regression model for drug response prediction."""

    @classmethod
    def get_model_name(cls) -> str:
        """Return the public model name."""
        return "SVR"


class GradientBoosting(SklearnModel):
    """Gradient boosting model for drug response prediction."""

    @classmethod
    def get_model_name(cls) -> str:
        """Return the public model name."""
        return "GradientBoosting"


class AdaBoostDecisionTree(SklearnModel):
    """AdaBoost model using decision-tree weak learners."""

    @classmethod
    def get_model_name(cls) -> str:
        """Return the public model name."""
        return "AdaBoostDecisionTree"


class LassoModel(SklearnModel):
    """Lasso regression model for drug response prediction."""

    @classmethod
    def get_model_name(cls) -> str:
        """Return the public model name."""
        return "Lasso"


class KNNRegressor(SklearnModel):
    """K-nearest-neighbor model for drug response prediction."""

    @classmethod
    def get_model_name(cls) -> str:
        """Return the public model name."""
        return "KNNRegressor"


__all__ = [
    "AdaBoostDecisionTree",
    "ElasticNetModel",
    "GradientBoosting",
    "KNNRegressor",
    "LassoModel",
    "RandomForest",
    "SingleDrugSklearnModel",
    "SklearnModel",
    "SVMRegressor",
]
