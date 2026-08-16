"""Tests that built-in component registration declares explicit contracts and metadata."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from drevalpy.components.contracts.contracts import FeatureContract
from drevalpy.registry._builtins import _registry_for, ensure_predictor_registered
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


class TestDiscoveryInAFreshProcess:
    """Extended tier: spawns an interpreter to prove registration needs no test setup.

    A class so the marker sits at class level; the in-process registration tests in
    this file stay in the fast tier.
    """

    pytestmark = pytest.mark.slow

    def test_fresh_process_discovery_returns_all_builtins(self) -> None:
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


class TestRegistryDispatchOnRescan:
    """``_registry_for`` recovers a class's registry from what registration stamped on it.

    This is the dispatch ``_reregister_from_module`` runs for every class in an
    already-imported module, which is how the registries refill after a ``clear()``.
    """

    def test_a_cell_line_featurizer_goes_back_to_the_cell_line_registry(self) -> None:
        from drevalpy.registry.cell_line_featurizer._registry import cell_line_featurizer_registry

        cls = get_cell_line_featurizer(list_cell_line_featurizers()[0])

        assert _registry_for(cls) is cell_line_featurizer_registry

    def test_a_drug_featurizer_goes_back_to_the_drug_registry(self) -> None:
        from drevalpy.registry.drug_featurizer._registry import drug_featurizer_registry

        cls = get_drug_featurizer(list_drug_featurizers()[0])

        assert _registry_for(cls) is drug_featurizer_registry

    def test_a_predictor_goes_back_to_the_predictor_registry(self) -> None:
        from drevalpy.registry.predictor._registry import predictor_registry

        assert _registry_for(get_predictor("naiveMean")) is predictor_registry

    def test_the_side_takes_precedence_over_a_predictor_contract(self) -> None:
        """A class carrying both is a featurizer: ``side`` is only ever stamped by a featurizer registry."""
        from drevalpy.registry.drug_featurizer._registry import drug_featurizer_registry

        class Ambiguous:
            side = "drug"
            cell_line_contract = object()

        assert _registry_for(Ambiguous) is drug_featurizer_registry

    def test_a_sideless_featurizer_falls_back_to_the_cell_line_registry(self) -> None:
        from drevalpy.registry.cell_line_featurizer._registry import cell_line_featurizer_registry

        class Sideless:
            contract = object()

        assert _registry_for(Sideless) is cell_line_featurizer_registry

    def test_a_plain_class_belongs_to_no_registry(self) -> None:
        class NotAComponent:
            pass

        assert _registry_for(NotAComponent) is None
