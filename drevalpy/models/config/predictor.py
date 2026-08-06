"""Declarative predictor configuration schema."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator, model_validator

from drevalpy.components.predictor_config_parse import normalize_predictor_config
from drevalpy.models.config.immutable import freeze_value, thaw_value


class PredictorConfig(BaseModel):
    """Immutable template for a predictor in a model stack."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    hyperparameter_space: Mapping[str, Any] | None = None

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
        if isinstance(data, dict) and "hyperparameters" in data:
            return normalize_predictor_config(data)
        return data

    @field_serializer("hyperparameter_space", when_used="always")
    def _serialize_hyperparameter_space(self, value: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        dumped = thaw_value(value)
        return dumped if isinstance(dumped, dict) else dict(dumped)

    @model_validator(mode="after")
    def _freeze_nested_mappings(self) -> PredictorConfig:
        if self.hyperparameter_space is not None:
            from drevalpy.components.hyperparameter_space import validate_hyperparameter_space

            validate_hyperparameter_space(
                self.hyperparameter_space,
                context=f"PredictorConfig({self.name!r}).hyperparameter_space",
            )
        object.__setattr__(
            self,
            "hyperparameter_space",
            freeze_value(self.hyperparameter_space) if self.hyperparameter_space is not None else None,
        )
        return self

    def create_instance(self, hyperparameters: Mapping[str, Any] | None = None):
        """Instantiate the configured predictor from the registry.

        :param hyperparameters: Concrete constructor values for this predictor.
        :returns: Predictor instance for this config.
        """
        from drevalpy.components.registry import get_predictor

        cls = get_predictor(self.name)
        hp = thaw_value(dict(hyperparameters or {}))
        return cls(hyperparameters=hp)
