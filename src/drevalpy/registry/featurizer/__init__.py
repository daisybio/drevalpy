"""The featurizer registry: ``register``, the ``featurizers`` registry and the side-checking lookup."""

from ._base import check_side, featurizers, get_featurizer, names_for_side, register

__all__ = ["check_side", "featurizers", "get_featurizer", "names_for_side", "register"]
