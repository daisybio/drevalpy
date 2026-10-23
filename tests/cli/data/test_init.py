"""Tests for the :mod:`drevalpy.cli.data` command group."""

from __future__ import annotations

from drevalpy.cli.data import data_app
from drevalpy.cli.data.concat import concat_datasets
from drevalpy.cli.data.load import load_dataset
from drevalpy.cli.data.split import split_dataset
from drevalpy.cli.data.synthetic import synthetic_dataset
from tests.cli._helpers import GroupSurface


class TestGroup(GroupSurface):
    group = data_app
    path = "data"
    callbacks = {
        "concat": concat_datasets,
        "load": load_dataset,
        "split": split_dataset,
        "synthetic": synthetic_dataset,
    }
