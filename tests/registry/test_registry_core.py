"""Tests for the explicit built-in registration and the registries built on ``Registry``."""

from __future__ import annotations

import pytest

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers._featurizer_label import requires_explicit_view
from drevalpy.registry._all import all_registries, restore_all, snapshot_all
from drevalpy.registry._builtins import BUILTIN_MODULES, builtin_names, get_skipped_builtin_modules, register_builtins
from drevalpy.registry._component import featurizer_component_metadata
from drevalpy.registry._registry import TableReprMixin
from drevalpy.registry.dataset import dataset_registry
from drevalpy.registry.featurizer import featurizer_registries
from drevalpy.registry.predictor import predictor_registry
from drevalpy.registry.splitter import splitter_registry
from drevalpy.registry.visualization import visualization_registry
from tests._trusted_subprocess import run_trusted_python


@pytest.mark.parametrize(
    "registry",
    [dataset_registry, splitter_registry, visualization_registry, predictor_registry],
    ids=["dataset", "splitter", "visualization", "predictor"],
)
def test_every_registry_shares_the_table_repr(registry: TableReprMixin) -> None:
    assert isinstance(registry, TableReprMixin)
    assert "<table" in registry._repr_html_()


def test_component_tables_list_name_description_and_tags() -> None:
    frame = predictor_registry.to_dataframe()

    assert list(frame.columns) == ["Name", "Description", "Tags"]
    assert frame["Name"].tolist() == list(predictor_registry.names())


def test_every_builtin_module_imports() -> None:
    assert get_skipped_builtin_modules() == {}


def test_register_builtins_is_idempotent() -> None:
    before = {kind: registry.names() for kind, registry in all_registries().items()}

    register_builtins()

    assert {kind: registry.names() for kind, registry in all_registries().items()} == before


def test_builtin_modules_are_unique() -> None:
    assert len(set(BUILTIN_MODULES)) == len(BUILTIN_MODULES)


def test_builtin_names_follow_the_order_of_builtin_modules() -> None:
    for registry in featurizer_registries.values():
        assert builtin_names(registry)[:3] == ("constant", "oneHot", "concatFeaturizers")
    assert builtin_names(predictor_registry)[0] == "naiveMeanEffects"
    assert builtin_names(predictor_registry)[-1] == "neuralNetwork"
    assert builtin_names(splitter_registry) == ("LDO", "LSO", "LPO", "LStO", "LTO")


def test_builtin_names_leave_out_other_registrations() -> None:
    @predictor_registry.decorator("notBuiltin")
    class NotBuiltin:
        pass

    assert "notBuiltin" in predictor_registry
    assert "notBuiltin" not in builtin_names(predictor_registry)


def test_restore_all_undoes_registrations_in_every_registry() -> None:
    snapshots = snapshot_all()
    predictor_registry.register("transient", object)
    splitter_registry.register("TRANSIENT", object)

    restore_all(snapshots)

    assert "transient" not in predictor_registry
    assert "TRANSIENT" not in splitter_registry


def test_builtin_featurizers_report_their_side_and_default_blocks() -> None:
    for side, registry in featurizer_registries.items():
        for name, cls in registry.items():
            meta = featurizer_component_metadata("featurizer", name, cls)
            assert meta["side"] == cls.side == side, name
            if not requires_explicit_view(name, side):
                assert all(isinstance(block.format, FeatureFormat) for block in cls.output_blocks({})), name


def test_bpe_pharmaformer_has_literature_reference() -> None:
    meta = featurizer_component_metadata(
        "featurizer", "bpePharmaformer", featurizer_registries["drug"].get("bpePharmaformer")
    )

    assert meta["repo_url"] == "https://github.com/zhouyuru1205/PharmaFormer"
    assert meta["citation_doi"] == "10.1038/s41698-025-01082-6"
    assert meta["citation"].startswith("https://doi.org/10.1038/")
    assert meta["deviations"]


@pytest.mark.slow
def test_fresh_process_registers_all_builtins() -> None:
    script = """
from drevalpy.registry.featurizer import names_for_side
from drevalpy.registry.predictor import predictor_registry
assert len(names_for_side("sample")) == 18
assert len(names_for_side("drug")) == 12
assert len(predictor_registry) == 31
print("ok")
"""
    completed = run_trusted_python(script)

    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout


@pytest.mark.parametrize(
    "name", ["precily", "srmf", "molir", "superfeltr", "pharmaFormer", "dipk", "sparsego", "naiveMean", "naiveDrugMean"]
)
def test_split_module_predictors_register_under_their_name(name: str) -> None:
    assert name in predictor_registry
