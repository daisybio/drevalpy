"""Named, unfitted model recipes shared by users and experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from drevalpy.models.config import ModelConfig
from drevalpy.models.config.resolved import ResolvedModelConfig
from drevalpy.models.tuning.public_flat import (
    apply_public_hyperparameters_to_config,
    public_hyperparameters_from_config,
)
from drevalpy.models.tuning.search_space import merge_model_config_spaces, resolve_model_config


@dataclass(frozen=True)
class ModelSpec:
    """A public model name and immutable, unfitted component configuration.

    Use :meth:`from_name` for a zoo preset, or supply a ``ModelConfig`` for a
    custom model. A specification contains no fitted state and can be reused
    across independent models, folds, and tuning trials.
    """

    name: str
    config: ModelConfig

    def __post_init__(self) -> None:
        """Validate identity and take an independent configuration snapshot."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ModelSpec.name must be a non-empty model name")
        if not isinstance(self.config, ModelConfig):
            raise TypeError("ModelSpec.config must be a ModelConfig")
        object.__setattr__(self, "config", ModelConfig.model_validate(self.config.model_dump(mode="python")))

    @classmethod
    def from_name(cls, name: str) -> ModelSpec:
        """Resolve a built-in or external zoo preset by its public name.

        :param name: Registered zoo preset name.
        :returns: An unfitted, reusable named recipe.
        """
        from drevalpy.models.zoo import zoo_model_config

        return cls(name, zoo_model_config(name))

    def resolve(self, hyperparameters: dict[str, Any] | None = None) -> ResolvedModelConfig:
        """Resolve defaults and optional public hyperparameters for one fit.

        :param hyperparameters: Component parameter overrides.
        :returns: Concrete immutable configuration.
        """
        return apply_public_hyperparameters_to_config(self.config, hyperparameters or {})

    @property
    def default_hyperparameters(self) -> dict[str, Any]:
        """Return independent public parameter defaults."""
        return public_hyperparameters_from_config(resolve_model_config(self.config))

    @property
    def hyperparameter_space(self) -> dict[str, Any]:
        """Return the qualified component search space."""
        return merge_model_config_spaces(self.config)

    @property
    def requires_early_stopping(self) -> bool:
        """Whether any configured component requires a stopping holdout."""
        from drevalpy.components.featurizers._featurizer_tree import iter_featurizer_leaves
        from drevalpy.registry.drug_featurizer import get as get_drug
        from drevalpy.registry.predictor import get as get_predictor
        from drevalpy.registry.sample_featurizer import get as get_sample

        if get_predictor(self.config.predictor.name).requires_early_stopping:
            return True
        for root, registry in (
            (self.config.sample_featurizer, get_sample),
            (self.config.drug_featurizer, get_drug),
        ):
            if root is not None and any(
                registry(leaf.name).requires_early_stopping for leaf in iter_featurizer_leaves(root, root.registry)
            ):
                return True
        return False

    @property
    def supports_early_stopping(self) -> bool:
        """Whether this recipe uses optional or required early stopping."""
        from drevalpy.registry.predictor import get as get_predictor

        return bool(get_predictor(self.config.predictor.name).supports_early_stopping) or self.requires_early_stopping
