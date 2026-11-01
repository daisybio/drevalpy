"""The featurizer registry: ``register``, the ``featurizers`` registry and the side-checking lookup."""

from ._base import featurizers, get_featurizer, names_for_side, register

__all__ = ["featurizers", "get_featurizer", "names_for_side", "register"]
