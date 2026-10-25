"""Construct fresh components from concrete configuration; no fitted state lives here."""

from __future__ import annotations

from typing import Any

from drevalpy.components.featurizers._featurizer_label import qualified_featurizer_selector
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.models.config import FeaturizerConfig
from drevalpy.models.config.resolved import ResolvedModelConfig


def _instantiate_featurizer(
    config: FeaturizerConfig,
    resolved: ResolvedModelConfig,
) -> Featurizer:
    registry = str(config.registry)
    if config.name == "concatFeaturizers":
        children = [_instantiate_featurizer(child, resolved) for child in (config.featurizers or ())]
        return config.create_instance({"featurizers": children})
    selector = qualified_featurizer_selector(config.name, config.view)
    return config.create_instance(resolved.featurizer_values(registry, selector))


def build_components(config: ResolvedModelConfig) -> tuple[Featurizer | None, Featurizer | None, Predictor]:
    """Instantiate featurizers and predictor for a validated config.

    :param config: Resolved model configuration.
    :returns: Fresh sample featurizer, optional drug featurizer, and predictor.
    """
    resolved = config
    template = resolved.template
    sample = (
        _instantiate_featurizer(template.sample_featurizer, resolved)
        if template.sample_featurizer is not None
        else None
    )
    drug = _instantiate_featurizer(template.drug_featurizer, resolved) if template.drug_featurizer is not None else None
    # A plain string: networks pass their hyperparameters to Lightning's ``save_hyperparameters``, and a checkpoint
    # holding the enum cannot be reloaded with ``torch.load(weights_only=True)``.
    predictor_hp: dict[str, Any] = {
        **resolved.predictor_values(),
        "prediction_mode": str(template.prediction_mode.value),
    }
    predictor = template.predictor.create_instance(predictor_hp)
    return sample, drug, predictor
