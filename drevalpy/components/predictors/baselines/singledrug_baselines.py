"""Concrete single-drug sklearn baseline adapters."""

from drevalpy.components.predictors.baselines.sklearn_base import SingleDrugSklearnModel


class SingleDrugElasticNet(SingleDrugSklearnModel):
    """ElasticNet model fitted independently for each drug."""

    @classmethod
    def get_model_name(cls) -> str:
        """Return the public model name."""
        return "SingleDrugElasticNet"


class SingleDrugRandomForest(SingleDrugSklearnModel):
    """Random forest model fitted independently for each drug."""

    @classmethod
    def get_model_name(cls) -> str:
        """Return the public model name."""
        return "SingleDrugRandomForest"
