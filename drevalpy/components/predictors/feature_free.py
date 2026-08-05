"""Feature-free predictors that consume only response values."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.predictors.base import Predictor


class FeatureFreePredictor(Predictor):
    """Predictors that do not consume featurizer outputs or raw feature datasets."""

    input_interface: ClassVar[str] = "feature_free"
    requires_drug_featurizer: ClassVar[bool] = False
