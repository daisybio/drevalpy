"""Validation of resolved hyperparameter mappings against a model config.

``resolved.py`` calls into this module during construction, so it keeps its
references to ``ModelConfig`` and ``FeaturizerConfig`` annotation-only. The key
grammar itself lives in the dependency-free
:mod:`drevalpy.models._hp_key_grammar` leaf, shared with
``drevalpy.models.tuning``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from drevalpy.components.featurizers._featurizer_tree import iter_featurizer_leaves
from drevalpy.models._hp_key_grammar import (
    featurizer_prefix,
    predictor_prefix,
    reject_indexed_featurizer_key,
)
from drevalpy.models.config._accepted_keys import featurizer_accepted_keys, leaf_selector, predictor_accepted_keys
from drevalpy.registry.predictor import get as get_predictor

if TYPE_CHECKING:
    from drevalpy.models.config.model import ModelConfig


def _build_validation_index(config: ModelConfig) -> dict[str, Any]:
    """Build the set of accepted qualified keys for a model config.

    Returns a dict mapping qualified key -> True for O(1) membership checks.
    """
    accepted: dict[str, Any] = {}

    predictor_cls = get_predictor(config.predictor.name)
    for param in predictor_accepted_keys(predictor_cls):
        accepted[predictor_prefix(config.predictor.name, param)] = True

    if config.sample_featurizer is not None:
        for leaf in iter_featurizer_leaves(config.sample_featurizer, "sample"):
            selector = leaf_selector(leaf)
            for param in featurizer_accepted_keys(leaf, "sample"):
                accepted[featurizer_prefix("sample", selector, param)] = True

    if config.drug_featurizer is not None:
        for leaf in iter_featurizer_leaves(config.drug_featurizer, "drug"):
            selector = leaf_selector(leaf)
            for param in featurizer_accepted_keys(leaf, "drug"):
                accepted[featurizer_prefix("drug", selector, param)] = True

    return accepted


def validate_merged_mapping(config: ModelConfig, merged: dict[str, Any]) -> None:
    """Reject unknown or malformed qualified hyperparameter keys.

    :param config: The model configuration template.
    :param merged: Merged mapping of qualified keys to values.
    :raises ValueError: If a key is not accepted by the model stack.
    """
    accepted = _build_validation_index(config)
    for key in merged:
        reject_indexed_featurizer_key(key)
        if key not in accepted:
            msg = f"Unknown hyperparameter {key!r} for this model stack."
            raise ValueError(msg)
