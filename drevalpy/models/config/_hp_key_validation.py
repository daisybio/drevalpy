"""Validation of resolved hyperparameter mappings against a model config.

This module exists to break the import cycle between ``resolved.py`` and
``hyperparameter_keys.py``.  It must NOT import ``ModelConfig``,
``FeaturizerConfig``, or anything from ``drevalpy.models.config`` or
``drevalpy.models.tuning`` at the top level.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from drevalpy.components.featurizers._featurizer_label import qualified_featurizer_selector
from drevalpy.components.featurizers._featurizer_tree import iter_featurizer_leaves
from drevalpy.components.registry import get_cell_line_featurizer, get_drug_featurizer, get_predictor

if TYPE_CHECKING:
    from drevalpy.models.config.featurizer import FeaturizerConfig
    from drevalpy.models.config.model import ModelConfig

_CELL_LINE_SLOT = "cell_line_featurizer"
_DRUG_SLOT = "drug_featurizer"

_REGISTRY_TO_SLOT = {
    "cell_line": _CELL_LINE_SLOT,
    "drug": _DRUG_SLOT,
}

_INDEXED_FEATURIZER_KEY_RE = re.compile(
    r"^(?P<slot>cell_line_featurizer|drug_featurizer)\."
    r"(?P<name>[^.]+)\.(?P<index>\d+)\.(?P<param>.+)$"
)


def _reject_indexed_featurizer_key(key: str) -> None:
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


def _featurizer_prefix(registry: str, selector: str, param: str) -> str:
    slot = _REGISTRY_TO_SLOT[registry]
    return f"{slot}.{selector}.{param}"


def _predictor_prefix(name: str, param: str) -> str:
    return f"predictor.{name}.{param}"


def _leaf_selector(featurizer: FeaturizerConfig) -> str:
    return qualified_featurizer_selector(featurizer.name, featurizer.view)


def _predictor_accepted_keys(predictor_cls: type[Any]) -> set[str]:
    keys = set(predictor_cls.get_default_hyperparameters())
    keys.update(predictor_cls.get_hyperparameter_space())
    non_tunable = getattr(predictor_cls, "non_tunable_hyperparameters", None)
    if isinstance(non_tunable, dict):
        keys.update(non_tunable)
    elif isinstance(non_tunable, (set, frozenset, list, tuple)):
        keys.update(str(key) for key in non_tunable)
    return keys


def _featurizer_accepted_keys(featurizer: FeaturizerConfig, registry: str) -> set[str]:
    cls = get_cell_line_featurizer(featurizer.name) if registry == "cell_line" else get_drug_featurizer(featurizer.name)
    return set(cls.get_hyperparameter_space())


def _build_validation_index(config: ModelConfig) -> dict[str, Any]:
    """Build the set of accepted qualified keys for a model config.

    Returns a dict mapping qualified key -> True for O(1) membership checks.
    """
    accepted: dict[str, Any] = {}

    predictor_cls = get_predictor(config.predictor.name)
    for param in _predictor_accepted_keys(predictor_cls):
        accepted[_predictor_prefix(config.predictor.name, param)] = True

    if config.cell_line_featurizer is not None:
        for leaf in iter_featurizer_leaves(config.cell_line_featurizer, "cell_line"):
            selector = _leaf_selector(leaf)
            for param in _featurizer_accepted_keys(leaf, "cell_line"):
                accepted[_featurizer_prefix("cell_line", selector, param)] = True

    if config.drug_featurizer is not None:
        for leaf in iter_featurizer_leaves(config.drug_featurizer, "drug"):
            selector = _leaf_selector(leaf)
            for param in _featurizer_accepted_keys(leaf, "drug"):
                accepted[_featurizer_prefix("drug", selector, param)] = True

    return accepted


def validate_merged_mapping(config: ModelConfig, merged: dict[str, Any]) -> None:
    """Reject unknown or malformed qualified hyperparameter keys.

    :param config: The model configuration template.
    :param merged: Merged mapping of qualified keys to values.
    :raises ValueError: Raised on invalid input.
    """
    accepted = _build_validation_index(config)
    for key in merged:
        _reject_indexed_featurizer_key(key)
        if key not in accepted:
            msg = f"Unknown hyperparameter {key!r} for this model stack."
            raise ValueError(msg)
