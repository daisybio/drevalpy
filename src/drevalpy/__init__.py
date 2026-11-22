"""Module containing the drevalpy suite."""

from importlib.metadata import version

from . import registry as registry
from ._experiment import experiment as experiment
from ._run import run as run
from ._train import train as train
from .data import split as split
from .data import synthetic as synthetic
from .data.datasets import load as load
from .execution import randomization as randomization
from .execution import robustness as robustness
from .models import DRPModel as DRPModel
from .models import ModelSpec as ModelSpec

__version__ = version("drevalpy")
