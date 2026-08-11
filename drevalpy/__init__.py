"""Module containing the drevalpy suite."""

from importlib.metadata import version

from . import registry as registry
from .data import split as split
from .data.datasets import load as load
from .experiment import randomization as randomization
from .experiment import robustness as robustness
from .models import construct_model as construct_model
from .run import run as run
from .single import single as single

__version__ = version("drevalpy")
