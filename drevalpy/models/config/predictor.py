"""Declarative predictor configuration schema."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from drevalpy.components.predictor_config_parse import normalize_predictor_config


class PredictorConfig(BaseModel):
    """Declarative specification for a predictor."""

    model_config = ConfigDict(extra="forbid")

    name: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    hyperparameter_space: dict[str, dict[str, Any]] | None = None

    @field_validator("name", mode="before")
    @classmethod
    def _coerce_name(cls, value: object) -> object:
        return str(value) if value is not None else value

    @model_validator(mode="before")
    @classmethod
    def _coerce_shorthand(cls, data: object) -> object:
        if isinstance(data, str):
            return normalize_predictor_config(data)
        if isinstance(data, dict) and "name" not in data:
            return normalize_predictor_config(data)
        return data

    def create_instance(self):
        """Instantiate the configured predictor from the registry.

        :returns: Predictor instance for this config.
        """
        from drevalpy.components.registry import get_predictor

        cls = get_predictor(self.name)
        return cls(hyperparameters=dict(self.hyperparameters))
