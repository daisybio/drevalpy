"""Resolved per-instance model configuration with concrete hyperparameter values."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from drevalpy.models.config.immutable import freeze_value, thaw_value
from drevalpy.models.config.model import ModelConfig


class ResolvedModelConfig(BaseModel):
    """Immutable instance config: template plus concrete qualified values.

    The ``template`` is the class-level ``ModelConfig``. ``values`` holds
    fully resolved concrete hyperparameters keyed by qualified names such as
    ``predictor.elasticNet.alpha`` or
    ``cell_line_featurizer.pca[expression].n_components``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    template: ModelConfig
    values: Mapping[str, Any] = Field(default_factory=dict)

    @field_serializer("values", when_used="always")
    def _serialize_values(self, value: Mapping[str, Any]) -> dict[str, Any]:
        dumped = thaw_value(value)
        return dumped if isinstance(dumped, dict) else dict(dumped)

    @model_validator(mode="after")
    def _freeze_and_validate_values(self) -> ResolvedModelConfig:
        object.__setattr__(self, "values", freeze_value(dict(self.values)))
        from drevalpy.components.tuning.hyperparameter_keys import validate_merged_mapping

        validate_merged_mapping(self.template, dict(self.values))
        return self

    @property
    def predictor_name(self) -> str:
        """Return the template predictor name.

        :returns: Predictor registry name.
        """
        return self.template.predictor.name

    def predictor_values(self) -> dict[str, Any]:
        """Return concrete local predictor hyperparameters.

        :returns: Mapping of predictor-local parameter names to values.
        """
        prefix = f"predictor.{self.template.predictor.name}."
        return {key.removeprefix(prefix): value for key, value in self.values.items() if key.startswith(prefix)}

    def featurizer_values(self, registry: str, selector: str) -> dict[str, Any]:
        """Return concrete local hyperparameters for one featurizer leaf.

        :param registry: ``cell_line`` or ``drug``.
        :param selector: Qualified featurizer selector (for example ``pca[expression]``).
        :returns: Mapping of featurizer-local parameter names to values.
        """
        slot = "cell_line_featurizer" if registry == "cell_line" else "drug_featurizer"
        prefix = f"{slot}.{selector}."
        return {key.removeprefix(prefix): value for key, value in self.values.items() if key.startswith(prefix)}
