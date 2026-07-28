"""Predictors that consume raw FeatureDataset views rather than featurizer outputs."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.predictors.base import Predictor


class RawDatasetPredictor(Predictor):
    """Predictor that reads original per-entity FeatureDataset views.

    Configured featurizers are forbidden. Concrete subclasses declare
    ``required_cell_line_views`` and ``required_drug_views``.
    """

    requires_drug_featurizer: ClassVar[bool] = False
    required_cell_line_views: ClassVar[tuple[str, ...]] = ()
    required_drug_views: ClassVar[tuple[str, ...]] = ()

    def active_cell_line_views(self) -> tuple[str, ...]:
        """Return cell-line views required for the current hyperparameters."""
        return self.required_cell_line_views

    def active_drug_views(self) -> tuple[str, ...]:
        """Return drug views required for the current hyperparameters."""
        return self.required_drug_views
