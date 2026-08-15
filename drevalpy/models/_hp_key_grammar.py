"""The string grammar of qualified hyperparameter keys.

A qualified key names one knob on one component of a composed model stack:

* ``predictor.<name>.<param>``
* ``cell_line_featurizer.<selector>.<param>``
* ``drug_featurizer.<selector>.<param>``

where ``<selector>`` is a featurizer name optionally qualified by a view, as in
``pca[methylation]``.

This module is a deliberate leaf: it imports nothing from ``drevalpy``, so both
``drevalpy.models.config`` and ``drevalpy.models.tuning`` can build and parse
keys from a single definition instead of each keeping its own copy.
"""

from __future__ import annotations

import re

__all__ = [
    "CELL_LINE_SLOT",
    "DRUG_SLOT",
    "FEATURIZER_SLOTS",
    "PREDICTOR_SLOT",
    "REGISTRY_TO_SLOT",
    "SLOT_TO_REGISTRY",
    "featurizer_prefix",
    "is_featurizer_slot_key",
    "predictor_prefix",
    "reject_indexed_featurizer_key",
    "split_predictor_key",
    "split_prefixed_key",
]

CELL_LINE_SLOT = "cell_line_featurizer"
DRUG_SLOT = "drug_featurizer"
PREDICTOR_SLOT = "predictor"

#: The two featurizer slots, in the order stacks declare them.
FEATURIZER_SLOTS = (CELL_LINE_SLOT, DRUG_SLOT)

#: Registry name (as used by ``FeaturizerConfig.registry``) to slot name.
REGISTRY_TO_SLOT = {
    "cell_line": CELL_LINE_SLOT,
    "drug": DRUG_SLOT,
}

SLOT_TO_REGISTRY = {slot: registry for registry, slot in REGISTRY_TO_SLOT.items()}

_SLOT_ALTERNATION = "|".join(FEATURIZER_SLOTS)

_INDEXED_FEATURIZER_KEY_RE = re.compile(
    rf"^(?P<slot>{_SLOT_ALTERNATION})\.(?P<name>[^.]+)\.(?P<index>\d+)\.(?P<param>.+)$"
)

_QUALIFIED_FEATURIZER_KEY_RE = re.compile(
    rf"^(?P<slot>{_SLOT_ALTERNATION})\.(?P<selector>[^.]+(?:\[[^\]]+\])?)\.(?P<param>.+)$"
)


def featurizer_prefix(registry: str, selector: str, param: str) -> str:
    """Build the qualified key for a featurizer parameter.

    :param registry: Registry name, ``cell_line`` or ``drug``.
    :param selector: Featurizer name, optionally view-qualified.
    :param param: Parameter name.
    :returns: ``<slot>.<selector>.<param>``.
    """
    return f"{REGISTRY_TO_SLOT[registry]}.{selector}.{param}"


def predictor_prefix(name: str, param: str) -> str:
    """Build the qualified key for a predictor parameter.

    :param name: Registered predictor name.
    :param param: Parameter name.
    :returns: ``predictor.<name>.<param>``.
    """
    return f"{PREDICTOR_SLOT}.{name}.{param}"


def is_featurizer_slot_key(key: str) -> bool:
    """Report whether *key* is already addressed at a featurizer slot.

    :param key: Candidate hyperparameter key.
    :returns: ``True`` when *key* starts with one of :data:`FEATURIZER_SLOTS`.
    """
    return any(key.startswith(f"{slot}.") for slot in FEATURIZER_SLOTS)


def reject_indexed_featurizer_key(key: str) -> None:
    """Refuse the withdrawn ``<slot>.<name>.<index>.<param>`` notation.

    :param key: Candidate hyperparameter key.
    :raises ValueError: When *key* uses the indexed notation.
    """
    match = _INDEXED_FEATURIZER_KEY_RE.match(key)
    if match is None:
        return
    slot = match.group("slot")
    name = match.group("name")
    param = match.group("param")
    msg = (
        f"Indexed featurizer hyperparameter keys are no longer supported: {key!r}. "
        f"Use a qualified selector such as "
        f"'{slot}.{name}[<view>].{param}' "
        f"or '{slot}.{name}.{param}'."
    )
    raise ValueError(msg)


def split_prefixed_key(key: str) -> tuple[str, str, str] | None:
    """Parse ``<slot>.<selector>.<param>`` into registry, selector, and param.

    :param key: Qualified hyperparameter key from a flat config.
    :returns: ``(registry, selector, param)`` tuple, or ``None`` when unparsable.
    :raises ValueError: When *key* uses the withdrawn indexed notation.
    """
    reject_indexed_featurizer_key(key)
    match = _QUALIFIED_FEATURIZER_KEY_RE.match(key)
    if match is None:
        return None
    return SLOT_TO_REGISTRY[match.group("slot")], match.group("selector"), match.group("param")


def split_predictor_key(key: str) -> tuple[str, str] | None:
    """Parse ``predictor.<name>.<param>`` into predictor name and param.

    :param key: Qualified hyperparameter key.
    :returns: ``(predictor_name, param)`` tuple, or ``None`` when unparsable.
    """
    parts = key.split(".")
    if len(parts) < 3 or parts[0] != PREDICTOR_SLOT:
        return None
    predictor_name, *param_parts = parts[1:]
    if not param_parts:
        return None
    return predictor_name, ".".join(param_parts)
