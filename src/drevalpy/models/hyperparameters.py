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

from drevalpy.components.contracts.hyperparameter_space import admits

if TYPE_CHECKING:
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


def qualify(known: Mapping[str, Any], values: Mapping[str, Any], *, check_values: bool = True) -> dict[str, Any]:
    """Return *values* under qualified keys.

    Short, qualified and mixed keys are accepted.

    :param known: The model's search space, whose keys are the qualified keys it owns.
    :param values: Hyperparameter values keyed by short or qualified names.
    :param check_values: Whether each value must lie in its search-space entry (see :func:`check_value`).
    :returns: The same values keyed by qualified names, in input order.
    :raises ValueError: If a key is unknown or ambiguous, two keys name the same hyperparameter, or
        *check_values* is set and a value lies outside its search-space entry.
    """
    owners: dict[str, list[str]] = defaultdict(list)
    for qualified in known:
        owners[_param(qualified)].append(qualified)
    result: dict[str, Any] = {}
    given: dict[str, str] = {}
    for name, value in values.items():
        qualified = _resolve(name, known, owners)
        if qualified in given:
            raise ValueError(
                f"Duplicate hyperparameter assignment for {qualified!r} from {given[qualified]!r} and {name!r}."
            )
        given[qualified] = name
        if check_values:
            check_value(qualified, value, known)
        result[qualified] = value
    return result


def _resolve(name: str, known: Mapping[str, Any], owners: Mapping[str, list[str]]) -> str:
    """Return the qualified key that the short or qualified *name* refers to.

    :param name: A key as the user wrote it.
    :param known: The model's search space.
    :param owners: Short parameter name mapped to the qualified keys that have it.
    :returns: The qualified key.
    :raises ValueError: If *name* is unknown or ambiguous.
    """
    if name in known:
        return name
    if "." not in name and len(owners.get(name, ())) == 1:
        return owners[name][0]
    if "." not in name and name in owners:
        alternatives = ", ".join(repr(owner) for owner in owners[name])
        raise ValueError(f"Ambiguous hyperparameter {name!r}. Use one of: {alternatives}.")
    raise ValueError(_unknown(name, known))


def validate(known: Mapping[str, Any], values: Mapping[str, Any]) -> None:
    """Check that every key of *values* is a qualified key of the model.

    Values are not checked against the space: this validates stored values, which
    ``with_hyperparameters(..., check_values=False)`` may have placed outside it.

    :param known: The model's search space, whose keys are the qualified keys it owns.
    :param values: Hyperparameter values keyed by qualified names.
    :raises ValueError: If a key is not a qualified key of the model.
    """
    for name in values:
        if name not in known:
            raise ValueError(_unknown(name, known))


def check_value(qualified: str, value: Any, known: Mapping[str, Any]) -> None:
    """Require *value* to be one tuning could draw for *qualified*; ``fixed`` entries take any value.

    :param qualified: A qualified key of *known*.
    :param value: The value given for it.
    :param known: The model's search space.
    :raises ValueError: If *value* lies outside the entry's range or choices.
    """
    spec = known[qualified]
    if not admits(spec, value):
        raise ValueError(
            f"Hyperparameter {qualified!r} = {value!r} is outside its search space ({_describe(spec)}). Widen "
            "the space with hyperparameter_space in the model config, or pass check_values=False to "
            "with_hyperparameters or DRPModel to use it anyway."
        )


def _describe(spec: Mapping[str, Any]) -> str:
    """Phrase the values a search-space entry admits."""
    kind = spec.get("type", "categorical")
    if kind == "categorical":
        return f"one of {list(spec['choices'])}"
    low, high = spec.get("low"), spec.get("high")
    if kind == "pow2":
        kind = "a power of two"
        low, high = (None if bound is None else 2 ** int(bound) for bound in (low, high))
    bounds = [f"{word} {bound}" for word, bound in (("from", low), ("to", high)) if bound is not None]
    return " ".join([kind, *bounds])


def _unknown(name: str, known: Mapping[str, Any]) -> str:
    """Phrase the error for a key no component of the model owns."""
    options = ", ".join(repr(qualified) for qualified in known) or "none"
    return f"Unknown hyperparameter {name!r} for this model; its hyperparameters are: {options}."
