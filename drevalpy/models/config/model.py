"""Full declarative model configuration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from drevalpy.components.predictor_config_parse import normalize_predictor_config
from drevalpy.types.model_scope import ModelScope
from drevalpy.types.prediction_mode import PredictionMode

from .featurizer import CellLineFeaturizerConfig, DrugFeaturizerConfig, FeaturizerConfig
from .predictor import PredictorConfig
from .single_drug import normalize_single_drug_identity


def _as_cell_line_featurizer_config(value: object) -> CellLineFeaturizerConfig:
    if isinstance(value, CellLineFeaturizerConfig):
        return value
    if isinstance(value, FeaturizerConfig):
        return CellLineFeaturizerConfig.model_validate(value.model_dump())
    return CellLineFeaturizerConfig.model_validate(value)


def _as_drug_featurizer_config(value: object) -> DrugFeaturizerConfig:
    if isinstance(value, DrugFeaturizerConfig):
        return value
    if isinstance(value, FeaturizerConfig):
        return DrugFeaturizerConfig.model_validate(value.model_dump())
    return DrugFeaturizerConfig.model_validate(value)


class ModelConfig(BaseModel):
    """Full declarative specification for a composed model."""

    model_config = ConfigDict(extra="forbid")

    cell_line_featurizer: CellLineFeaturizerConfig | None = None
    drug_featurizer: DrugFeaturizerConfig | None = None
    predictor: PredictorConfig
    prediction_mode: PredictionMode = PredictionMode.REGRESSION
    scope: ModelScope = ModelScope.MULTI_DRUG

    @model_validator(mode="before")
    @classmethod
    def _normalize_sections(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        cell_line = normalized.get("cell_line_featurizer")
        if cell_line is not None and not isinstance(cell_line, CellLineFeaturizerConfig):
            if isinstance(cell_line, (str, list, dict, FeaturizerConfig)):
                normalized["cell_line_featurizer"] = _as_cell_line_featurizer_config(cell_line)
        drug = normalized.get("drug_featurizer")
        if drug is not None and not isinstance(drug, DrugFeaturizerConfig):
            if isinstance(drug, (str, list, dict, FeaturizerConfig)):
                normalized["drug_featurizer"] = _as_drug_featurizer_config(drug)
        predictor = normalized.get("predictor")
        if predictor is not None and not isinstance(predictor, PredictorConfig):
            if isinstance(predictor, (str, dict)):
                normalized["predictor"] = normalize_predictor_config(predictor)
        return normalize_single_drug_identity(normalized)

    @property
    def model_id(self) -> str | None:
        """Stable identifier for a fully specified combination.

        :returns: Colon-separated featurizer and predictor names, or ``None`` when incomplete.
        """
        if self.cell_line_featurizer is None and self.drug_featurizer is None:
            return self.predictor.name
        if self.cell_line_featurizer is None:
            return None
        if (
            self.scope == ModelScope.SINGLE_DRUG
            and self.drug_featurizer is not None
            and self.drug_featurizer.name == "identity"
        ):
            return f"{self.cell_line_featurizer.name}:{self.predictor.name}"
        if self.drug_featurizer is None:
            return None
        return f"{self.cell_line_featurizer.name}:" f"{self.drug_featurizer.name}:" f"{self.predictor.name}"
