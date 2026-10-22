"""Concatenate datasets of different studies into one ordinary dataset."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from drevalpy.types.data.dataset import Dataset


def concat(
    datasets: Sequence[Dataset],
    *,
    join: Literal["exact", "inner"] = "exact",
    name: str | None = None,
) -> Dataset:
    """Concatenate datasets, one per study, into one :class:`~drevalpy.types.data.dataset.Dataset`.

    The result is an ordinary dataset, so every function that takes one (``split``,
    ``run``, ``aggregate``) works on it unchanged:

    * Each row is one (study, sample) with the ID ``<study>::<sample_id>``, where the
      study is the input's ``name``. ``obs`` gains the columns ``study`` and
      ``source_sample_id``. A sample measured in two studies gives two rows, so no
      measurement is lost.
    * Sample views are stacked row-wise, view by view.
    * Drug columns are the union of the drug IDs. Response layers are the union, with
      NaN where a study did not measure a pair.
    * ``uns["dataset_name"]`` is *name*, or the input names joined by ``+``.

    The inputs are not modified.

    Args:
        datasets: At least two datasets with distinct names, none of them randomized.
        join: ``"exact"`` requires every input to carry the same feature names and
            units in each view they share. ``"inner"`` keeps only the feature names
            common to all inputs. A unit mismatch raises with either value.
        name: Name of the result. Defaults to the input names joined by ``+``,
            such as ``"GDSC2+CTRPv2"``.

    Returns:
        The concatenated dataset.

    Raises:
        ValueError: If fewer than two datasets are given, names are not unique, an input
            is randomized, feature names or units differ between inputs, or one drug ID
            has two different SMILES strings.
    """
    raise NotImplementedError("W1-P1")
