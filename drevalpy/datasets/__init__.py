"""
Module for handling datasets.
"""

__all__ = ["GDSC1", "GDSC2", "CCLE", "Toy", "RESPONSE_DATASET_FACTORY"]
from .gdsc1 import GDSC1
from .gdsc2 import GDSC2
from .ccle import CCLE
from .toy import Toy

RESPONSE_DATASET_FACTORY = {
    "GDSC1": GDSC1,
    "GDSC2": GDSC2,
    "CCLE": CCLE,
    "Toy_Data": Toy,
}
