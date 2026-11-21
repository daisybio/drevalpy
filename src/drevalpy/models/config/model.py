"""Full declarative model configuration template."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator, model_validator

from drevalpy.models.config._parse import format_featurizer_recipe, format_model_recipe
from drevalpy.models.config.featurizer import FeaturizerConfig
from drevalpy.models.config.predictor import PredictorConfig
from drevalpy.models.config.validation import validate
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.types.data.feature_source import Side
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

    @field_validator("sample_featurizer", "drug_featurizer", mode="before")
    @classmethod
    def _side_featurizer_slots(cls, value: object, info: ValidationInfo) -> object:
        """Read a featurizer slot for its side, so its nodes look up that side's registry.

        A field validator, so pydantic reports errors of the slot under its name.

        :param value: The slot's featurizer config in any accepted spelling, or ``None``.
        :param info: Validation info naming the slot.
        :returns: The slot validated for its side.
        """
        if value is None:
            return value
        side = Side.SAMPLE if info.field_name == "sample_featurizer" else Side.DRUG
        return FeaturizerConfig.parse(value, side)

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
        """The model recipe of this config, which ``from_recipe`` reads back into the same components.

        Each featurizer slot is written with its views and concat children
        (``view[gene_expression]+view[mutations]:fingerprints:elasticNet``); fixed options and
        search spaces are not part of a recipe. A drug featurizer without a sample featurizer
        cannot be expressed as one.

        :returns: The recipe, or ``None`` for a drug-only stack.
        """
        sample, drug = (
            None if slot is None else format_featurizer_recipe(slot)
            for slot in (self.sample_featurizer, self.drug_featurizer)
        )
        if sample is None and drug is not None:
            return None
        return format_model_recipe(sample, drug, self.predictor.name)

    def sample_views(self) -> list[str]:
        """Return the raw view names required by the sample featurizer tree.

        :returns: View names required by the sample featurizer tree.
        """
        return [] if self.sample_featurizer is None else self.sample_featurizer.views()

    def drug_views(self) -> list[str]:
        """Return the raw view names required by the drug featurizer tree.

        :returns: View names required by the drug featurizer tree.
        """
        return [] if self.drug_featurizer is None else self.drug_featurizer.views()
