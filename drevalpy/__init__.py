"""Module containing the drevalpy suite."""

from importlib.metadata import version

from . import registry as registry
from ._run import run as run
from ._single import single as single
from .data import split as split
from .data.datasets import load as load
from .experiment import randomization as randomization
from .experiment import robustness as robustness
from .models import construct_model as construct_model

__version__ = version("drevalpy")
