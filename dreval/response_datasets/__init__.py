__all__ = [
    "GDSC1", "GDSC2"
]
from .gdsc1 import GDSC1
from .gdsc2 import GDSC2

RESPONSE_DATASET_FACTORY = {"GDSC1": GDSC1, "GDSC2": GDSC2}
