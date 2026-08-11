"""Declarative predictor configuration schema."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from drevalpy.models.config._predictor_parse import normalize_predictor_config
from drevalpy.models.config.immutable import FrozenMapping, thaw_value


class PredictorConfig(BaseModel):
    """Immutable template for a predictor in a model stack."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    hyperparameter_space: FrozenMapping | None = None

    @field_validator("name", mode="before")
    @classmethod
    def _coerce_name(cls, value: object) -> object:
        return str(value) if value is not None else value

    @model_validator(mode="before")
    @classmethod
    def _normalize_recipe_input(cls, data: object) -> object:
        """Rewrite the various ways of writing a predictor into this model's own fields.

        The featurizer counterpart of this hook, for the predictor slot: accepts a bare
        recipe string such as ``"elasticNet"`` or a one-key mapping like
        ``{"randomForest": {"n_estimators": 10}}``, and reduces each to ``name`` plus
        ``hyperparameter_space``.

        :param data: A recipe string or a mapping of fields.
        :returns: Canonical field mapping, or *data* unchanged if it is already canonical.
        """
        if isinstance(data, str):
            return normalize_predictor_config(data)
        if isinstance(data, dict) and "name" not in data:
            return normalize_predictor_config(data)
        return data

    @model_validator(mode="after")
    def _validate_hyperparameter_space(self) -> PredictorConfig:
        if self.hyperparameter_space is not None:
            from drevalpy.components.contracts.hyperparameter_space import validate_hyperparameter_space

            validate_hyperparameter_space(
                self.hyperparameter_space,
                context=f"PredictorConfig({self.name!r}).hyperparameter_space",
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
