"""Declarative predictor configuration schema."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from drevalpy.components.hyperparameter_space import effective_space, validate_hyperparameter_space
from drevalpy.models.config._parse import normalize_predictor
from drevalpy.models.config.immutable import FrozenMapping, thaw_value
from drevalpy.registry.predictor import get as get_predictor

if TYPE_CHECKING:
    from drevalpy.components.predictors.abstract.base import Predictor


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
        """Accept a recipe string or a ``defaults`` mapping in place of the canonical fields.

        :param data: A recipe string such as ``"elasticNet"`` or a mapping of fields.
        :returns: Canonical field mapping, or *data* unchanged for any other type.
        """
        if isinstance(data, (str, dict)):
            return normalize_predictor(data)
        return data

    @model_validator(mode="after")
    def _validate_hyperparameter_space(self) -> PredictorConfig:
        if self.hyperparameter_space is not None:
            validate_hyperparameter_space(
                self.hyperparameter_space,
                context=f"PredictorConfig({self.name!r}).hyperparameter_space",
            )
        return self

    @property
    def effective_space(self) -> dict[str, Any]:
        """The search space this predictor is tuned over: its declared ``hyperparameter_space``, else its class's."""
        return effective_space(self.hyperparameter_space, get_predictor(self.name))

    def create_instance(self, hyperparameters: Mapping[str, Any] | None = None) -> Predictor:
        """Instantiate the configured predictor from the registry.

        :param hyperparameters: Concrete constructor values for this predictor.
        :returns: Predictor instance for this config.
        """
        cls = get_predictor(self.name)
        hp = thaw_value(dict(hyperparameters or {}))
        return cls(hyperparameters=hp)
