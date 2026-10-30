"""Full declarative model configuration template."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from drevalpy.models.config._recipe import format_model_recipe
from drevalpy.models.config.featurizer import FeaturizerConfig
from drevalpy.models.config.predictor import PredictorConfig
from drevalpy.models.config.validation import validate
from drevalpy.models.config.view_resolution import views_from_featurizer_config
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.types.enums.model_scope import ModelScope
from drevalpy.types.enums.prediction_mode import PredictionMode


class ModelConfig(BaseModel):
    """Immutable class-level template for a composed model.

    Stores architecture (featurizers / predictor), prediction mode, and optional
    hyperparameter-space overrides. Concrete hyperparameter values live on
    :class:`~drevalpy.models.spec.ModelSpec`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_featurizer: FeaturizerConfig | None = None
    drug_featurizer: FeaturizerConfig | None = None
    predictor: PredictorConfig
    prediction_mode: PredictionMode = PredictionMode.REGRESSION

    _validate_semantics = model_validator(mode="after")(validate)

    @property
    def scope(self) -> ModelScope:
        """Whether this stack trains one model per drug or one across all drugs.

        Not a choice a config makes: single-drug variants are separate registered predictors,
        so the scope follows from the predictor and the two can never disagree.

        :returns: The scope declared by the configured predictor.
        """
        return get_predictor(self.predictor.name).scope

    @property
    def model_id(self) -> str | None:
        """Stable identifier for a fully specified combination.

        The identifier is a model recipe, so it is written by the same code that reads one.
        A drug featurizer without a sample featurizer cannot be expressed as a recipe.

        :returns: Colon-separated featurizer and predictor names, or ``None`` for a drug-only stack.
        """
        if self.sample_featurizer is None:
            return self.predictor.name if self.drug_featurizer is None else None
        drug = None if self.drug_featurizer is None else self.drug_featurizer.name
        return format_model_recipe(self.sample_featurizer.name, drug, self.predictor.name)

    def sample_views(self) -> list[str]:
        """Return the raw view names required by the sample featurizer tree.

        :returns: View names required by the sample featurizer tree.
        """
        if self.sample_featurizer is None:
            return []
        return views_from_featurizer_config(self.sample_featurizer)

    def drug_views(self) -> list[str]:
        """Return the raw view names required by the drug featurizer tree.

        :returns: View names required by the drug featurizer tree.
        """
        if self.drug_featurizer is None:
            return []
        return views_from_featurizer_config(self.drug_featurizer)
