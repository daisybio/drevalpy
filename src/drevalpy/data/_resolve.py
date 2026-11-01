"""Turn the dataset argument of ``run`` and the CLI (names, paths, datasets, or several of them) into one dataset."""

from __future__ import annotations

from collections.abc import Sequence

from drevalpy.errors import InvalidInputError
from drevalpy.types.data.dataset import Dataset

from ._concat import concat
from .datasets._load import load


def resolve_dataset(dataset: Dataset | str | Sequence[Dataset | str]) -> Dataset:
    """Load names and paths with :func:`~drevalpy.data.load`, and concatenate several datasets into one.

    :param dataset: A dataset, a registered name or ``.h5mu`` path, or a sequence of them.
    :returns: The dataset; for a sequence of two or more, their :func:`~drevalpy.data.concat`.
    :raises InvalidInputError: If *dataset* is an empty sequence.
    """
    if isinstance(dataset, str) or not isinstance(dataset, Sequence):
        return load(dataset) if isinstance(dataset, str) else dataset
    if not dataset:
        raise InvalidInputError("dataset must not be an empty sequence")
    datasets = [load(item) if isinstance(item, str) else item for item in dataset]
    return datasets[0] if len(datasets) == 1 else concat(datasets)
