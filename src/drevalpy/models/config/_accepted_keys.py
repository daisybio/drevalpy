"""Which hyperparameter keys a model stack's components accept.

Shared by key validation (``_hp_key_validation``) and by the ownership index and search-space
code in ``drevalpy.models.tuning``, which all need the same answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from drevalpy.components.featurizers._featurizer_label import qualified_featurizer_selector
from drevalpy.registry.featurizer import Side, get_featurizer

if TYPE_CHECKING:
    from drevalpy.components.predictors.abstract.base import Predictor
    from drevalpy.models.config.featurizer import FeaturizerConfig


def leaf_selector(featurizer: FeaturizerConfig) -> str:
    """Return the selector a featurizer leaf carries in qualified hyperparameter keys.

    :param featurizer: Leaf featurizer config.
    :returns: Name, qualified by the view when the featurizer is view-parametric, e.g. ``pca[methylation]``.
    """
    return qualified_featurizer_selector(featurizer.name, featurizer.view)


def predictor_accepted_keys(predictor_cls: type[Predictor]) -> set[str]:
    """Return every hyperparameter key a predictor class accepts.

    :param predictor_cls: Registered predictor class.
    :returns: Defaults, tunable space keys and the ``non_tunable_hyperparameters`` keys.
    """
    keys = set(predictor_cls.get_default_hyperparameters())
    keys.update(predictor_cls.get_hyperparameter_space())
    keys.update(getattr(predictor_cls, "non_tunable_hyperparameters", {}))
    return keys


def featurizer_accepted_keys(featurizer: FeaturizerConfig, side: Side) -> set[str]:
    """Return the tunable hyperparameter keys of a featurizer's registered class.

    :param featurizer: Leaf featurizer config.
    :param side: Entity side the featurizer is registered on.
    :returns: Keys of the class's declared hyperparameter space.
    """
    return set(get_featurizer(side, featurizer.name).get_hyperparameter_space())
