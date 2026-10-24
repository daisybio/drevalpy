"""Tests that built-in component registration declares explicit contracts and metadata."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from drevalpy.components.contracts.contracts import FeatureContract
from drevalpy.registry._builtins import _registry_for
from drevalpy.registry.drug_featurizer import get as get_drug_featurizer
from drevalpy.registry.drug_featurizer import list as list_drug_featurizers
from drevalpy.registry.drug_featurizer import metadata as get_drug_featurizer_metadata
from drevalpy.registry.featurizer import drug_featurizer_registry, sample_featurizer_registry
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.registry.predictor._registry import predictor_registry
from drevalpy.registry.sample_featurizer import get as get_sample_featurizer
from drevalpy.registry.sample_featurizer import list as list_sample_featurizers
from tests._trusted_subprocess import run_trusted_python
from tests.registry._helpers import restore_component_registries


@pytest.fixture(autouse=True)
def _register_components() -> Iterator[None]:
    restore_component_registries()
    yield
    restore_component_registries()


@pytest.mark.parametrize(
    ("names", "get"),
    [(list_sample_featurizers, get_sample_featurizer), (list_drug_featurizers, get_drug_featurizer)],
    ids=["sample", "drug"],
)
def test_builtin_featurizers_declare_contract(names, get) -> None:
    for name in names():
        cls = get(name)
        assert "contract" in cls.__dict__, name
        assert isinstance(cls.contract, FeatureContract)


def test_bpe_pharmaformer_has_literature_reference() -> None:
    meta = get_drug_featurizer_metadata("bpePharmaformer")

    assert meta["repo_url"] == "https://github.com/zhouyuru1205/PharmaFormer"
    assert meta["citation_doi"] == "10.1038/s41698-025-01082-6"
    assert meta["citation"].startswith("https://doi.org/10.1038/")
    assert meta["deviations"]


@pytest.mark.slow
def test_fresh_process_discovery_returns_all_builtins() -> None:
    script = """
from drevalpy.registry.sample_featurizer import sample_featurizer_registry
from drevalpy.registry.drug_featurizer import drug_featurizer_registry
from drevalpy.registry.predictor import predictor_registry
assert len(sample_featurizer_registry.list_metadata()) == 18
assert len(drug_featurizer_registry.list_metadata()) == 12
assert len(predictor_registry.list_metadata()) == 30
print("ok")
"""
    completed = run_trusted_python(script)

    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout


@pytest.mark.parametrize(
    "name", ["precily", "srmf", "molir", "superfeltr", "pharmaFormer", "dipk", "sparsego", "naiveMean", "naiveDrugMean"]
)
def test_split_module_predictors_register_under_their_name(name: str) -> None:
    assert get_predictor(name).registry_name == name


class TestRegistryDispatchOnRescan:
    """``_registry_for`` recovers a class's registry from what registration stamped on it."""

    def test_registered_components_go_back_to_their_own_registry(self) -> None:
        assert _registry_for(get_sample_featurizer(list_sample_featurizers()[0])) is sample_featurizer_registry
        assert _registry_for(get_drug_featurizer(list_drug_featurizers()[0])) is drug_featurizer_registry
        assert _registry_for(get_predictor("naiveMean")) is predictor_registry

    def test_the_side_takes_precedence_over_a_predictor_contract(self) -> None:
        class Ambiguous:
            side = "drug"
            sample_contract = object()

        assert _registry_for(Ambiguous) is drug_featurizer_registry

    def test_a_sideless_featurizer_falls_back_to_the_sample_registry(self) -> None:
        class Sideless:
            contract = object()

        assert _registry_for(Sideless) is sample_featurizer_registry

    def test_a_plain_class_belongs_to_no_registry(self) -> None:
        assert _registry_for(type("NotAComponent", (), {})) is None
