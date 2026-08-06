"""Full declarative model configuration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from drevalpy.types.model_scope import ModelScope
from drevalpy.types.prediction_mode import PredictionMode

from .featurizer import CellLineFeaturizerConfig, DrugFeaturizerConfig
from .predictor import PredictorConfig


class ModelConfig(BaseModel):
    """Full declarative specification for a composed model."""

    model_config = ConfigDict(extra="forbid")

    cell_line_featurizer: CellLineFeaturizerConfig | None = None
    drug_featurizer: DrugFeaturizerConfig | None = None
    predictor: PredictorConfig
    prediction_mode: PredictionMode = PredictionMode.REGRESSION
    scope: ModelScope = ModelScope.MULTI_DRUG

    @model_validator(mode="after")
    def _inject_single_drug_identity(self) -> ModelConfig:
        """Fill ``drug_featurizer: identity`` for single-drug feature stacks that omit it.

        :returns: This config, with identity injected when applicable.
        """
        if (
            self.scope == ModelScope.SINGLE_DRUG
            and self.cell_line_featurizer is not None
            and self.drug_featurizer is None
        ):
            self.drug_featurizer = DrugFeaturizerConfig(name="identity")
        return self

    @property
    def model_id(self) -> str | None:
        """Stable identifier for a fully specified combination.

        :returns: Colon-separated featurizer and predictor names, or ``None`` when incomplete.
        """
        cell = self.cell_line_featurizer
        drug = self.drug_featurizer
        if cell is None:
            return self.predictor.name if drug is None else None
        if drug is None:
            return None
        parts = [cell.name]
        if self.scope != ModelScope.SINGLE_DRUG or drug.name != "identity":
            parts.append(drug.name)
        parts.append(self.predictor.name)
        return ":".join(parts)
