"""Full declarative model configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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

    def validate(self) -> None:  # type: ignore[override]  # supported public API; not pydantic model_validate
        """Check registry slots, feature compatibility, and prediction mode."""
        from drevalpy.models.config.validation import validate_model_config

        normalized = normalize_single_drug_identity(self.model_dump())
        if normalized != self.model_dump():
            refreshed = ModelConfig.model_validate(normalized)
            self.drug_featurizer = refreshed.drug_featurizer
            self.scope = refreshed.scope
        validate_model_config(self)

    @classmethod
    def from_spec(
        cls,
        spec: str,
        *,
        hyperparameters: dict[str, Any] | None = None,
        prediction_mode: PredictionMode = PredictionMode.REGRESSION,
    ) -> ModelConfig:
        """Build a config from a recipe, zoo, legacy, or baseline spec string.

        :param spec: Zoo preset name, colon-separated recipe, or legacy baseline token.
        :param hyperparameters: Optional flat public hyperparameter overrides.
        :param prediction_mode: Regression or classification mode for the predictor.
        :returns: Validated ``ModelConfig`` instance.
        """
        from drevalpy.models.config.spec import build_model_config_from_spec

        return build_model_config_from_spec(
            spec,
            hyperparameters=hyperparameters,
            prediction_mode=prediction_mode,
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> ModelConfig:
        """Load a config from a YAML file.

        :param path: Path to a YAML mapping describing the model config.
        :returns: Validated ``ModelConfig`` instance.
        """
        from drevalpy.models.config.io import model_config_from_yaml

        return model_config_from_yaml(path)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        """Build a config from a plain dictionary.

        :param data: Mapping with featurizer and predictor sections.
        :returns: Validated ``ModelConfig`` instance.
        """
        from drevalpy.models.config.io import model_config_from_dict

        return model_config_from_dict(data)
