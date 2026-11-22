"""Feature-free predictors that consume only response values."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.inputs import Inputs


class FeatureFreePredictor(Predictor):
    """Predictors that do not consume featurizer outputs or raw feature datasets."""

    sample_inputs: ClassVar[Inputs] = Inputs.none()
    drug_inputs: ClassVar[Inputs] = Inputs.none()
