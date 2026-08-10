"""Module containing the drevalpy suite."""

from importlib.metadata import version

from .components.registry.featurizer_registry import (
    cell_line_featurizer_registry as cell_line_featurizer_registry,
)
from .components.registry.featurizer_registry import (
    drug_featurizer_registry as drug_featurizer_registry,
)
from .components.registry.predictor_registry import predictor_registry as predictor_registry
from .data import dataset_registry as dataset_registry
from .data import split as split
from .data import splitter_registry as splitter_registry
from .data.datasets import load as load
from .experiment import randomization as randomization
from .experiment import run as run
from .experiment import robustness as robustness
from .models import construct_model as construct_model
from .pipeline import pipeline as pipeline

__version__ = version("drevalpy")
