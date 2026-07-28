"""Raw FeatureDataset literature engine adapter."""

from __future__ import annotations

from typing import ClassVar, cast

import numpy as np

from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.literature._engine_mixin import LiteratureEngineMixin
from drevalpy.components.predictors.literature._raw_views import (
    validate_pyg_drug_graphs,
    validate_required_views,
)
from drevalpy.components.predictors.raw_dataset import RawDatasetPredictor
from drevalpy.datasets.dataset import FeatureDataset


class RawLiteratureEnginePredictor(LiteratureEngineMixin, RawDatasetPredictor):
    """Train a literature engine on raw FeatureDataset views only."""

    requires_drug_featurizer: ClassVar[bool] = False
    required_cell_line_views: ClassVar[tuple[str, ...]] = ()
    required_drug_views: ClassVar[tuple[str, ...]] = ()
    validate_drug_graphs: ClassVar[bool] = False

    def _validate_inputs(
        self,
        cell_line_input: FeatureDataset | None,
        drug_input: FeatureDataset | None,
    ) -> tuple[FeatureDataset, FeatureDataset | None]:
        name = getattr(self, "registry_name", self.__class__.__name__)
        cell_views = self.active_cell_line_views()
        drug_views = self.active_drug_views()
        validate_required_views(
            cell_line_input,
            cell_views,
            predictor_name=str(name),
            side="cell_line",
        )
        cell_lines = cast(FeatureDataset, cell_line_input)
        drugs: FeatureDataset | None = None
        if drug_views:
            validate_required_views(
                drug_input,
                drug_views,
                predictor_name=str(name),
                side="drug",
            )
            drugs = cast(FeatureDataset, drug_input)
            if self.validate_drug_graphs and "drug_graph" in drug_views:
                validate_pyg_drug_graphs(drugs, predictor_name=str(name))
        return cell_lines, drugs

    def fit(self, batch: ModelInputBatch) -> None:
        cell_lines, drugs = self._validate_inputs(batch.cell_line_input, batch.drug_input)
        self._train_engine(batch, cell_lines, drugs)

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        cell_lines, drugs = self._validate_inputs(batch.cell_line_input, batch.drug_input)
        return self._predict_engine(batch, cell_lines, drugs)
