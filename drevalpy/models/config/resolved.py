"""Resolved per-instance model configuration with concrete hyperparameter values."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from drevalpy.models.config.immutable import FrozenMapping
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
    values: FrozenMapping = Field(default_factory=dict, validate_default=True)

    @model_validator(mode="after")
    def _validate_values(self) -> ResolvedModelConfig:
        from drevalpy.models.tuning.hyperparameter_keys import validate_merged_mapping

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

    def cell_line_views(self) -> list[str]:
        """Return the raw view names required by the cell-line featurizer tree.

        :returns: View names required by the cell-line featurizer tree.
        """
        return self.template.cell_line_views(resolved=self)

    def drug_views(self) -> list[str]:
        """Return the raw view names required by the drug featurizer tree.

        :returns: View names required by the drug featurizer tree.
        """
        return self.template.drug_views(resolved=self)
