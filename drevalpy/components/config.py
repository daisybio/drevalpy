"""Declarative configuration for modular featurizer/predictor pairing."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PredictionMode(StrEnum):
    """Whether the model predicts a continuous response or a discrete class."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class FeaturizerConfig(BaseModel):
    """Declarative specification for a featurizer."""

    model_config = ConfigDict(extra="forbid")

    type: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    registry: Literal["cell_line", "drug"] = "cell_line"
    view: str | None = None
    views: list[str] | None = None
    hyperparameter_space: dict[str, dict[str, Any]] | None = None

    def create_instance(self):
        """Instantiate the configured featurizer from the registry."""
        from drevalpy.components.registry import lookup as reg

        if self.registry == "cell_line":
            cls = reg.get_cell_line_featurizer(self.type)
        else:
            cls = reg.get_drug_featurizer(self.type)
        hp = dict(self.hyperparameters)
        if self.view is not None:
            hp.setdefault("view", self.view)
        if self.views is not None:
            hp.setdefault("views", self.views)
        return cls(**hp)


class PredictorConfig(BaseModel):
    """Declarative specification for a predictor."""

    model_config = ConfigDict(extra="forbid")

    type: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    hyperparameter_space: dict[str, dict[str, Any]] | None = None

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_type(cls, value: object) -> object:
        return str(value) if value is not None else value

    @model_validator(mode="before")
    @classmethod
    def _coerce_shorthand(cls, data: object) -> object:
        if isinstance(data, str):
            return {"type": data}
        return data

    def create_instance(self):
        """Instantiate the configured predictor from the registry."""
        from drevalpy.components.registry import lookup as reg

        cls = reg.get_predictor(self.type)
        return cls()


class ModelConfig(BaseModel):
    """Full declarative specification for a composed model."""

    model_config = ConfigDict(extra="forbid")

    cell_line_featurizer: FeaturizerConfig | None = None
    drug_featurizer: FeaturizerConfig | None = None
    predictor: PredictorConfig
    prediction_mode: PredictionMode = PredictionMode.REGRESSION

    @model_validator(mode="before")
    @classmethod
    def _normalize_sections(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        cell_line = normalized.get("cell_line_featurizer")
        if isinstance(cell_line, dict):
            cell_line_payload = dict(cell_line)
            cell_line_payload.setdefault("registry", "cell_line")
            normalized["cell_line_featurizer"] = cell_line_payload
        drug = normalized.get("drug_featurizer")
        if isinstance(drug, dict):
            drug_payload = dict(drug)
            drug_payload.setdefault("registry", "drug")
            normalized["drug_featurizer"] = drug_payload
        return normalized

    @property
    def model_id(self) -> str | None:
        """Stable identifier for a fully specified combination."""
        if self.cell_line_featurizer is None and self.drug_featurizer is None:
            return self.predictor.type
        if self.cell_line_featurizer is None or self.drug_featurizer is None:
            return None
        return f"{self.cell_line_featurizer.type}:" f"{self.drug_featurizer.type}:" f"{self.predictor.type}"

    def validate(self) -> None:  # type: ignore[override]
        """Check registry slots, feature compatibility, and prediction mode."""
        from drevalpy.components.validation import validate_model_config

        validate_model_config(self)

    def create_model(self):
        """Build a runnable `~drevalpy.models.composed_model.ComposedModel`."""
        from drevalpy.models.composed_model import ComposedModel

        self.validate()
        cell_line = self.cell_line_featurizer.create_instance() if self.cell_line_featurizer else None
        drug = self.drug_featurizer.create_instance() if self.drug_featurizer else None
        pred = self.predictor.create_instance()
        return ComposedModel(
            cell_line,
            drug,
            pred,
            predictor_hyperparameters=self.predictor.hyperparameters,
            prediction_mode=self.prediction_mode,
        )

    @classmethod
    def from_spec(
        cls,
        spec: str,
        *,
        hyperparameters: dict[str, Any] | None = None,
        prediction_mode: PredictionMode = PredictionMode.REGRESSION,
    ) -> ModelConfig:
        """Build a config from a recipe, zoo, legacy, or baseline spec string."""
        from drevalpy.models.model_config_spec import build_model_config_from_spec

        return build_model_config_from_spec(
            spec,
            hyperparameters=hyperparameters,
            prediction_mode=prediction_mode,
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> ModelConfig:
        """Load a config from a YAML file."""
        from drevalpy.models.config_io import model_config_from_yaml

        return model_config_from_yaml(path)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        """Build a config from a plain dictionary."""
        from drevalpy.models.config_io import model_config_from_dict

        return model_config_from_dict(data)
