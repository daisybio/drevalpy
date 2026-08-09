"""Module containing the drevalpy suite."""

from importlib.metadata import version

from . import data as data
from .components.registry.featurizer_registry import (
    cell_line_featurizer_registry as cell_line_featurizer_registry,
)
from .components.registry.featurizer_registry import (
    drug_featurizer_registry as drug_featurizer_registry,
)
from .components.registry.predictor_registry import predictor_registry as predictor_registry
from .data import dataset_registry as dataset_registry
from .data import splitter_registry as splitter_registry

__version__ = version("drevalpy")
