"""Hyperparameter keys of a model: their grammar, resolving user input to them, and checking them.

Internally every hyperparameter is addressed by its qualified key, one knob on one component:

* ``predictor.<name>.<param>``
* ``sample_featurizer.<selector>.<param>``
* ``drug_featurizer.<selector>.<param>``

where ``<selector>`` is a featurizer name, qualified by its view when it takes one, as in
``pca[methylation]``. Results, W&B configs and checkpoints carry these keys.

The keys a model owns are the keys of its search space,
:func:`drevalpy.models.tuning.search_space.space`. Users may also write the short ``<param>``
when exactly one component of the model owns it; :func:`qualify` resolves both forms once, in
:meth:`drevalpy.models.spec.ModelSpec.with_hyperparameters`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from drevalpy.models.config import ModelConfig
    from drevalpy.types.data.feature_source import Side

PREDICTOR_SLOT = "predictor"

#: The featurizer slot of each side, in the order a model declares them.
FEATURIZER_SLOTS: dict[Side, str] = {"sample": "sample_featurizer", "drug": "drug_featurizer"}


def key(slot: str, selector: str, param: str) -> str:
    """Build the qualified key of one parameter of one component.

    :param slot: ``predictor``, ``sample_featurizer`` or ``drug_featurizer``.
    :param selector: Predictor name, or featurizer selector such as ``pca[methylation]``.
    :param param: Parameter name.
    :returns: ``<slot>.<selector>.<param>``.
    """
    return f"{slot}.{selector}.{param}"


def component_values(values: Mapping[str, Any], slot: str, selector: str) -> dict[str, Any]:
    """Return the values one component owns, under its own parameter names.

    :param values: Qualified hyperparameter values of a model.
    :param slot: Slot of the component.
    :param selector: Predictor name or featurizer selector of the component.
    :returns: Parameter name to value, in the order of *values*.
    """
    prefix = key(slot, selector, "")
    return {name.removeprefix(prefix): value for name, value in values.items() if name.startswith(prefix)}


def _param(qualified: str) -> str:
    """Return the short name of a qualified key: parameter names never contain a dot."""
    return qualified.rsplit(".", 1)[1]


def qualify(config: ModelConfig, values: Mapping[str, Any]) -> dict[str, Any]:
    """Return *values* under qualified keys.

    Short, qualified and mixed keys are accepted.

    :param config: Model configuration whose components own the hyperparameters.
    :param values: Hyperparameter values keyed by short or qualified names.
    :returns: The same values keyed by qualified names, in input order.
    :raises ValueError: If a key is unknown or ambiguous, or two keys name the same hyperparameter.
    """
    from drevalpy.models.tuning.search_space import space

    known = space(config)
    owners: dict[str, list[str]] = defaultdict(list)
    for qualified in known:
        owners[_param(qualified)].append(qualified)
    result: dict[str, Any] = {}
    given: dict[str, str] = {}
    for name, value in values.items():
        if name in known:
            qualified = name
        elif "." not in name and len(owners.get(name, ())) == 1:
            qualified = owners[name][0]
        elif "." not in name and name in owners:
            alternatives = ", ".join(repr(owner) for owner in owners[name])
            raise ValueError(f"Ambiguous hyperparameter {name!r}. Use one of: {alternatives}.")
        else:
            raise ValueError(_unknown(name, known))
        if qualified in given:
            raise ValueError(
                f"Duplicate hyperparameter assignment for {qualified!r} from {given[qualified]!r} and {name!r}."
            )
        given[qualified] = name
        result[qualified] = value
    return result


def validate(config: ModelConfig, values: Mapping[str, Any]) -> None:
    """Check that every key of *values* is a qualified key that *config*'s components own.

    :param config: Model configuration whose components own the hyperparameters.
    :param values: Hyperparameter values keyed by qualified names.
    :raises ValueError: If a key is not a qualified key of *config*.
    """
    from drevalpy.models.tuning.search_space import space

    known = space(config)
    for name in values:
        if name not in known:
            raise ValueError(_unknown(name, known))


def _unknown(name: str, known: Mapping[str, Any]) -> str:
    """Phrase the error for a key no component of the model owns."""
    options = ", ".join(repr(qualified) for qualified in known) or "none"
    return f"Unknown hyperparameter {name!r} for this model; its hyperparameters are: {options}."
