"""Named, unfitted model recipes shared by users and experiments."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from drevalpy.components.contracts.hyperparameter_space import default_values
from drevalpy.errors import UnknownNameError
from drevalpy.models import hyperparameters as hp
from drevalpy.models.config import ModelConfig
from drevalpy.models.config.immutable import freeze_value, thaw_value
from drevalpy.models.tuning.search_space import space


class UnknownModelError(UnknownNameError):
    """A model name that is neither a zoo name nor a valid recipe.

    :ivar name: The name as given.
    :ivar detail: Why reading *name* as a recipe failed, without the recipe itself repeated.
    """

    def __init__(self, name: str, detail: str) -> None:
        """Build the message naming both readings of *name*.

        :param name: The name as given.
        :param detail: Why the recipe reading failed.
        """
        super().__init__(f"{name!r} is not a zoo name and not a valid recipe: {detail}")
        self.name = name
        self.detail = detail


@dataclass(frozen=True)
class ModelSpec:
    """A public model name, its component configuration and one value per hyperparameter.

    Use :meth:`from_name` for a zoo preset or recipe, or supply a ``ModelConfig`` for a custom
    model. ``hyperparameters`` is keyed by qualified keys
    (:mod:`drevalpy.models.hyperparameters`) and always complete: a key left out takes the
    default of its space entry. :meth:`with_hyperparameters` derives a spec with other values,
    accepting short keys too. A spec holds no fitted state and can be reused across folds and
    tuning trials.
    """

    name: str
    config: ModelConfig
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the name and the qualified keys and fill in defaults.

        The config is used as given: a ``ModelConfig`` is validated when built and deep-frozen.

        :raises ValueError: If the name is empty or a hyperparameter key is not a qualified key of the config.
        :raises TypeError: If ``config`` is not a ``ModelConfig``.
        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ModelSpec.name must be a non-empty model name")
        if not isinstance(self.config, ModelConfig):
            raise TypeError("ModelSpec.config must be a ModelConfig")
        hp.validate(self.hyperparameter_space, self.hyperparameters)
        defaults = default_values(self.hyperparameter_space, context=f"hyperparameter space of {self.name!r}")
        object.__setattr__(self, "hyperparameters", freeze_value({**defaults, **self.hyperparameters}))

    @classmethod
    def from_name(cls, name: str) -> ModelSpec:
        """Resolve a zoo preset by its zoo name, or else read *name* as a recipe.

        A zoo name wins over a recipe reading. The spec is named by *name* exactly as given,
        so a recipe is reported under its full recipe string.

        :param name: Zoo name, or a recipe such as ``"pca[gene_expression]:fingerprints:elasticNet"``.
        :returns: An unfitted, reusable named recipe with default hyperparameters.
        :raises UnknownModelError: If *name* is neither a zoo name nor a valid recipe; ``detail``
            on the error says why the recipe reading failed.
        """
        from drevalpy.models.config.io import from_recipe
        from drevalpy.models.zoo import get_zoo_config

        try:
            return cls(name, get_zoo_config(name))
        except UnknownNameError:
            pass
        try:
            config = from_recipe(name)
        except ValueError as exc:
            raise UnknownModelError(name, str(exc)) from exc
        return cls(name, config)

    def with_hyperparameters(self, values: Mapping[str, Any], *, check_values: bool = True) -> ModelSpec:
        """Return this spec with some hyperparameters replaced.

        :param values: New values under short keys (``alpha``, when one component owns it) or
            qualified keys (``predictor.elasticNet.alpha``).
        :param check_values: Whether each value must lie in its search-space entry. ``False`` allows
            values tuning could not draw, such as ``max_epochs=1`` for a quick run; keys are still checked.
        :returns: A spec with the same name and config, and *values* over this spec's values. A key that
            is unknown or ambiguous, or a value outside its entry when *check_values* is set, raises
            ``ValueError``.
        """
        qualified = hp.qualify(self.hyperparameter_space, values, check_values=check_values)
        # A shallow copy keeps the cached search space; qualified keys need no second validation.
        derived = copy.copy(self)
        object.__setattr__(derived, "hyperparameters", freeze_value({**self.hyperparameters, **qualified}))
        return derived

    @cached_property
    def hyperparameter_space(self) -> dict[str, Any]:
        """The qualified search space of all components, computed once per spec; treat it as read-only."""
        return space(self.config)

    @property
    def model_id(self) -> str | None:
        """The recipe naming the components, as :attr:`ModelConfig.model_id`."""
        return self.config.model_id

    def dump(self) -> dict[str, Any]:
        """Return the spec as plain data, the inverse of :meth:`load`.

        :returns: ``name``, ``config`` (``model_dump(mode="json")``) and ``hyperparameters``.
        """
        return {
            "name": self.name,
            "config": self.config.model_dump(mode="json"),
            "hyperparameters": thaw_value(self.hyperparameters),
        }

    @classmethod
    def load(cls, data: Mapping[str, Any]) -> ModelSpec:
        """Rebuild a spec from :meth:`dump` output.

        :param data: Mapping with ``name``, ``config`` and ``hyperparameters``.
        :returns: The equivalent spec.
        """
        return cls(data["name"], ModelConfig.model_validate(data["config"]), data["hyperparameters"])

    @property
    def requires_early_stopping(self) -> bool:
        """Whether any configured component requires a stopping holdout."""
        from drevalpy.registry.featurizer import get_featurizer
        from drevalpy.registry.predictor import get as get_predictor

        if get_predictor(self.config.predictor.name).requires_early_stopping:
            return True
        for root in (self.config.sample_featurizer, self.config.drug_featurizer):
            if root is not None and any(get_featurizer(leaf.name).requires_early_stopping for leaf in root.leaves()):
                return True
        return False

    @property
    def supports_early_stopping(self) -> bool:
        """Whether this recipe uses optional or required early stopping."""
        from drevalpy.registry.predictor import get as get_predictor

        return bool(get_predictor(self.config.predictor.name).supports_early_stopping) or self.requires_early_stopping
