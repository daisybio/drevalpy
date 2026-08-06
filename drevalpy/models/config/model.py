"""Full declarative model configuration template."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from drevalpy.models.config.featurizer import CellLineFeaturizerConfig, DrugFeaturizerConfig
from drevalpy.models.config.immutable import rebuild_model
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

    @model_validator(mode="after")
    def _validate_semantics(self) -> ModelConfig:
        """Run registry / contract / block-schema checks once at construction.

        :returns: This validated config.
        """
        validate(self)
        return self

    @property
    def model_id(self) -> str | None:
        """Stable identifier for a fully specified combination.

        :returns: Colon-separated featurizer and predictor names, or ``None`` when incomplete.
        """
        cell = self.cell_line_featurizer
        drug = self.drug_featurizer
        if cell is None:
            return self.predictor.name if drug is None else None
        if drug is None:
            return None
        parts = [cell.name]
        if self.scope != ModelScope.SINGLE_DRUG or drug.name != "identity":
            parts.append(drug.name)
        parts.append(self.predictor.name)
        return ":".join(parts)

    def replace(self, **updates: Any) -> ModelConfig:
        """Return a validated copy with the given field updates.

        :param updates: Field overrides.
        :returns: Newly validated ``ModelConfig``.
        """
        return rebuild_model(self, **updates)
