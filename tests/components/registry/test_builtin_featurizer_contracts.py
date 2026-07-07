"""Tests that built-in featurizers declare explicit registration contracts."""

from __future__ import annotations

import pytest

from drevalpy.components.contracts import FeatureContract
from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.components.registry import (
    clear_cell_line_featurizer_registry,
    clear_drug_featurizer_registry,
    get_cell_line_featurizer,
    get_drug_featurizer,
    list_cell_line_featurizers,
    list_drug_featurizers,
)


@pytest.fixture(autouse=True)
def _register_components() -> None:
    clear_cell_line_featurizer_registry()
    clear_drug_featurizer_registry()
    register_builtin_components()
    yield
    clear_cell_line_featurizer_registry()
    clear_drug_featurizer_registry()
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


def test_bpe_pharmaformer_is_literature_component() -> None:
    from drevalpy.components.registry import get_drug_featurizer_metadata

    meta = get_drug_featurizer_metadata("bpePharmaformer")
    assert meta["category"] == "literature"
    assert meta["template_repo_url"] == "https://github.com/zhouyuru1205/PharmaFormer"
    assert meta["citation_doi"] == "10.1038/s41698-025-01082-6"
    assert meta["citation"].startswith("https://doi.org/10.1038/")
