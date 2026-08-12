"""Tests that built-in component registration declares explicit contracts and metadata."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from drevalpy.components.contracts.contracts import FeatureContract
from drevalpy.registry._builtins import ensure_predictor_registered
from drevalpy.registry.cell_line_featurizer import (
    get as get_cell_line_featurizer,
)
from drevalpy.registry.cell_line_featurizer import (
    list as list_cell_line_featurizers,
)
from drevalpy.registry.drug_featurizer import (
    get as get_drug_featurizer,
)
from drevalpy.registry.drug_featurizer import (
    list as list_drug_featurizers,
)
from drevalpy.registry.predictor import get as get_predictor
from tests._trusted_subprocess import run_trusted_python
from tests.registry._helpers import restore_component_registries


@pytest.fixture(autouse=True)
def _register_components() -> Iterator[None]:
    restore_component_registries()
    yield
    restore_component_registries()


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


def test_fresh_process_discovery_returns_all_builtins() -> None:
    script = """
from drevalpy.registry.cell_line_featurizer import cell_line_featurizer_registry
from drevalpy.registry.drug_featurizer import drug_featurizer_registry
from drevalpy.registry.predictor import predictor_registry
assert len(cell_line_featurizer_registry.list_metadata()) == 17
assert len(drug_featurizer_registry.list_metadata()) == 10
assert len(predictor_registry.list_metadata()) == 27
print("ok")
"""
    completed = run_trusted_python(script)
    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout


def test_literature_predictors_register_from_split_modules() -> None:
    for name in ("precily", "srmf", "molir", "superfeltr", "pharmaFormer", "dipk", "sparsego"):
        ensure_predictor_registered(name)
        cls = get_predictor(name)
        assert cls.registry_name == name


def test_naive_predictors_register_from_package() -> None:
    for name in ("naiveMean", "naiveDrugMean", "naiveMeanEffects"):
        ensure_predictor_registered(name)
        cls = get_predictor(name)
        assert cls.registry_name == name
