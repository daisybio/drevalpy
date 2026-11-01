"""Loading a registered dataset by name, or a dataset file by path, with :func:`load`.

The dataset registry it reads is :mod:`drevalpy.registry.dataset`.
"""

from ._load import load

__all__ = ["load"]
