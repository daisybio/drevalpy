"""Randomized-dataset generation used by the feature-importance tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from drevalpy.execution._randomization import randomization
from drevalpy.models import ModelSpec


@dataclass
class _StubConfig:
    samples: list[str] = field(default_factory=list)
    drugs: list[str] = field(default_factory=list)

    def featurizers(self) -> dict[str, SimpleNamespace]:
        return {
            "sample": SimpleNamespace(views=lambda: list(self.samples)),
            "drug": SimpleNamespace(views=lambda: list(self.drugs)),
        }


class _StubDataset:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.randomization_provenance: dict[str, Any] | None = None
        self.tag = "dataset"
        self.randomization: tuple[str, str] | None = None

    def with_randomized_views(self, views: list[str], **kwargs: Any) -> _StubDataset:
        self.calls.append((list(views), kwargs))
        child = _StubDataset()
        child.tag = f"dataset({','.join(views)})"
        child.randomization = kwargs["randomization"]
        child.randomization_provenance = {"effective_seed": kwargs["random_state"]}
        return child


def _stub_model(*, samples: list[str] | None = None, drugs: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(config=_StubConfig(samples=samples or [], drugs=drugs or []))


@pytest.mark.parametrize(
    ("mode", "model_kwargs", "expected"),
    [
        ("SVRC", {"samples": ["expr", "mut"]}, [["expr"], ["mut"]]),
        ("SVCC", {"samples": ["expr", "mut"]}, [["mut"], ["expr"]]),
        ("SVCC", {"samples": ["a", "b", "c"]}, [["b", "c"], ["a", "c"], ["a", "b"]]),
        ("SVRD", {"drugs": ["fp", "smiles"]}, [["fp"], ["smiles"]]),
        ("SVCD", {"drugs": ["fp", "smiles"]}, [["smiles"], ["fp"]]),
        ("NOPE", {"samples": ["expr"]}, []),
    ],
)
def test_modes_select_the_right_views(mode, model_kwargs, expected) -> None:
    dataset = _StubDataset()
    randomization(_stub_model(**model_kwargs), dataset, [mode])
    assert [views for views, _ in dataset.calls] == expected


def test_returns_one_tagged_dataset_per_test() -> None:
    dataset = _StubDataset()
    model = _stub_model(samples=["expr", "mut"], drugs=["fp"])
    results = randomization(model, dataset, ["SVRC", "SVRD"])
    assert [result.tag for result in results] == ["dataset(expr)", "dataset(mut)", "dataset(fp)"]
    assert [kwargs["randomization"] for _, kwargs in dataset.calls] == [
        ("SVRC", "expr"),
        ("SVRC", "mut"),
        ("SVRD", "fp"),
    ]
    assert randomization(model, _StubDataset(), []) == []


def test_type_and_seed_defaults_and_overrides() -> None:
    dataset = _StubDataset()
    result = randomization(_stub_model(samples=["expr"]), dataset, ["SVRC"])
    _, kwargs = dataset.calls[0]
    assert kwargs["randomization_type"] == "permutation"
    assert isinstance(kwargs["random_state"], int)
    assert result[0].randomization_provenance == {"root_seed": 42, "effective_seed": kwargs["random_state"]}

    dataset = _StubDataset()
    randomization(_stub_model(samples=["expr"]), dataset, ["SVRC"], randomization_type="invariant", random_state=7)
    assert dataset.calls[0][1]["randomization_type"] == "invariant"


def test_effective_seeds_are_stable_distinct_and_order_independent() -> None:
    def seeds(views: list[str]) -> dict[str, int]:
        dataset = _StubDataset()
        randomization(_stub_model(samples=views), dataset, ["SVRC"], random_state=7)
        return {kwargs["randomization"][1]: kwargs["random_state"] for _, kwargs in dataset.calls}

    forward = seeds(["expr", "mut"])
    assert forward == seeds(["expr", "mut"]) == seeds(["mut", "expr"])
    assert len(set(forward.values())) == 2


def test_an_empty_complement_is_skipped_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="drevalpy.execution._randomization"):
        results = randomization(_stub_model(samples=["expr"], drugs=["fp", "smiles"]), _StubDataset(), ["SVCC", "SVCD"])
    assert [result.randomization for result in results] == [("SVCD", "fp"), ("SVCD", "smiles")]
    assert "Skipping the SVCC randomization test of 'expr': no other view to randomize" in caplog.text


def test_real_model_config_views_and_real_dataset(synthetic_dataset) -> None:
    spec = ModelSpec.from_name("ElasticNet")
    stub = _StubDataset()
    randomization(spec, stub, ["SVRC", "SVRD"])
    assert [views for views, _ in stub.calls] == [["gene_expression"], ["morgan_fingerprint"]]

    results = randomization(spec, synthetic_dataset, ["SVRC"], random_state=0)
    assert [ds.randomization for ds in results] == [("SVRC", "gene_expression")]
    assert results[0] is not synthetic_dataset
    assert synthetic_dataset.randomization is None
    provenance = results[0].randomization_provenance
    assert provenance is not None
    assert (provenance["root_seed"], provenance["views"], provenance["type"]) == (0, ["gene_expression"], "permutation")


@pytest.mark.parametrize("model", ["PharmaFormer", "PaccMann"])
def test_a_model_computing_from_the_smiles_has_no_drug_view_to_randomize(synthetic_dataset, model: str) -> None:
    assert "bpe_smiles" in synthetic_dataset.mdata.mod["response"].varm

    assert randomization(ModelSpec.from_name(model), synthetic_dataset, ["SVRD", "SVCD"]) == []
