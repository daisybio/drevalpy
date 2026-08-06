"""Full declarative model configuration template."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from drevalpy.models.config._recipe import format_model_recipe
from drevalpy.models.config.featurizer import CellLineFeaturizerConfig, DrugFeaturizerConfig
from drevalpy.models.config.predictor import PredictorConfig
from drevalpy.models.config.validation import validate
from drevalpy.types.model_scope import ModelScope
from drevalpy.types.prediction_mode import PredictionMode


class ModelConfig(BaseModel):
    """Immutable class-level template for a composed model.

    Stores architecture (featurizers / predictor), prediction mode, scope, and
    optional hyperparameter-space overrides. Concrete selected hyperparameter
    values live on ``ResolvedModelConfig``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    cell_line_featurizer: CellLineFeaturizerConfig | None = None
    drug_featurizer: DrugFeaturizerConfig | None = None
    predictor: PredictorConfig
    prediction_mode: PredictionMode = PredictionMode.REGRESSION
    scope: ModelScope = ModelScope.MULTI_DRUG

    _validate_semantics = model_validator(mode="after")(validate)

    @model_validator(mode="before")
    @classmethod
    def _inject_single_drug_identity(cls, data: Any) -> Any:
        """Fill ``drug_featurizer: identity`` for single-drug stacks that omit it.

        Runs before Pydantic parses fields, so ``data`` may not be a dict and ``scope`` may still
        be a raw string (e.g. straight from YAML). Bad values are passed through untouched so that
        normal field validation reports the error later.

        :param data: Raw constructor / validation payload.
        :returns: Payload with identity drug featurizer injected when applicable.
        """
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        scope = payload.get("scope", ModelScope.MULTI_DRUG)
        try:
            scope = scope if isinstance(scope, ModelScope) else ModelScope(scope)
        except (TypeError, ValueError):
            return payload
        if (
            scope == ModelScope.SINGLE_DRUG
            and payload.get("cell_line_featurizer") is not None
            and payload.get("drug_featurizer") is None
        ):
            payload["drug_featurizer"] = DrugFeaturizerConfig(name="identity")
        return payload

    @property
    def model_id(self) -> str | None:
        """Stable identifier for a fully specified combination.

        The identifier is a model recipe, so it is written by the same code that reads one.
        A half-specified stack has no name: one featurizer without the other cannot be
        expressed as a recipe.

        :returns: Colon-separated featurizer and predictor names, or ``None`` when incomplete.
        """
        cell = self.cell_line_featurizer
        drug = self.drug_featurizer
        if cell is None:
            return self.predictor.name if drug is None else None
        if drug is None:
            return None
        # Single-drug stacks route per drug through the identity featurizer rather than
        # featurizing it, so naming it would suggest a choice the user never made.
        omit_drug = self.scope == ModelScope.SINGLE_DRUG and drug.name == "identity"
        return format_model_recipe(cell.name, None if omit_drug else drug.name, self.predictor.name)
