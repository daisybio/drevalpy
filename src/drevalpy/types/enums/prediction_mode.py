"""Prediction task mode for composed models and predictors."""

from __future__ import annotations

from enum import StrEnum


class PredictionMode(StrEnum):
    """Whether the model predicts a continuous response or a discrete class."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"
