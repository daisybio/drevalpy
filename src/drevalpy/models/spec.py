"""Named, unfitted model recipes shared by users and experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from drevalpy.models.config import ModelConfig
from drevalpy.models.config.resolved import ResolvedModelConfig
from drevalpy.models.tuning.public_flat import (
    apply_public_hyperparameters_to_config,
    public_hyperparameters_from_config,
)
from drevalpy.models.tuning.search_space import merge_model_config_spaces, resolve_model_config


class UnknownModelError(ValueError):
    """A model name that is neither a zoo name nor a valid recipe.

    Attributes:
        name: The name as given.
        detail: Why reading *name* as a recipe failed, without the recipe itself repeated.
    """

    def __init__(self, name: str, detail: str) -> None:
        """Build the message naming both readings of *name*.

        :param name: The name as given.
        :param detail: Why the recipe reading failed.
        """
        super().__init__(f"{name!r} is not a zoo name and not a valid recipe: {detail}")
        self.name = name
        self.detail = detail


def _recipe_error_detail(name: str, exc: ValueError) -> str:
    """Strip the wrapper text a recipe error repeats around its actual reason."""
    return str(exc).removeprefix(f"Invalid model config in recipe {name.strip()!r}: ").replace("Value error, ", "")


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
        """Resolve a zoo preset by its zoo name, or else read *name* as a recipe.

        A zoo name wins over a recipe reading. The spec is named by *name* exactly as given,
        so a recipe is reported under its full recipe string.

        :param name: Zoo name, or a recipe such as ``"pca[gene_expression]:fingerprints:elasticNet"``.
        :returns: An unfitted, reusable named recipe.
        :raises UnknownModelError: If *name* is neither a zoo name nor a valid recipe; ``detail``
            on the error says why the recipe reading failed.
        """
        from drevalpy.models.config.io import from_spec
        from drevalpy.models.zoo import zoo_model_config

        try:
            return cls(name, zoo_model_config(name))
        except KeyError:
            pass
        try:
            config = from_spec(name)
        except ValueError as exc:
            raise UnknownModelError(name, _recipe_error_detail(name, exc)) from exc
        return cls(name, cast("ModelConfig", config))

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
