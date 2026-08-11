"""Tests that built-in featurizers declare explicit registration contracts."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from drevalpy.components.contracts.contracts import FeatureContract
from drevalpy.registry._builtins import register_builtin_components
from drevalpy.registry.cell_line_featurizer import cell_line_featurizer_registry
from drevalpy.registry.cell_line_featurizer import (
    get as get_cell_line_featurizer,
)
from drevalpy.registry.cell_line_featurizer import (
    list as list_cell_line_featurizers,
)
from drevalpy.registry.drug_featurizer import drug_featurizer_registry
from drevalpy.registry.drug_featurizer import (
    get as get_drug_featurizer,
)
from drevalpy.registry.drug_featurizer import (
    list as list_drug_featurizers,
)


@pytest.fixture(autouse=True)
def _register_components() -> Iterator[None]:
    cell_line_featurizer_registry.clear()
    drug_featurizer_registry.clear()
    register_builtin_components()
    yield
    cell_line_featurizer_registry.clear()
    drug_featurizer_registry.clear()
    register_builtin_components()


def test_builtin_cell_line_featurizers_declare_contract() -> None:
    for name in list_cell_line_featurizers():
        cls = get_cell_line_featurizer(name)
        assert "contract" in cls.__dict__, name
        assert isinstance(cls.contract, FeatureContract)


def test_builtin_drug_featurizers_declare_contract() -> None:
    for name in list_drug_featurizers():
        cls = get_drug_featurizer(name)
        assert "contract" in cls.__dict__, name
        assert isinstance(cls.contract, FeatureContract)


def test_bpe_pharmaformer_has_literature_reference() -> None:
    from drevalpy.registry.drug_featurizer import metadata as get_drug_featurizer_metadata

    meta = get_drug_featurizer_metadata("bpePharmaformer")
    assert meta["repo_url"] == "https://github.com/zhouyuru1205/PharmaFormer"
    assert meta["citation_doi"] == "10.1038/s41698-025-01082-6"
    assert meta["citation"].startswith("https://doi.org/10.1038/")
    assert meta["deviations"]
