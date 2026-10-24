"""Randomized-dataset generation used by the feature-importance tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from drevalpy.data import synthetic
from drevalpy.experiment._randomization import (
    _complement_view_tests,
    _single_view_tests,
    randomization,
)
from drevalpy.models import ModelSpec


@dataclass
class _StubConfig:
    samples: list[str] = field(default_factory=list)
    drugs: list[str] = field(default_factory=list)

    def sample_views(self) -> list[str]:
        return list(self.samples)

    def drug_views(self) -> list[str]:
        return list(self.drugs)


class _StubDataset:
    def __init__(self, missing: tuple[str, ...] = ()) -> None:
        self.missing = missing
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.mdata = SimpleNamespace(uns={})
        self.tag = "dataset"
        self.randomization: tuple[str, str] | None = None

    def with_randomized_views(self, views: list[str], **kwargs: Any) -> _StubDataset:
        self.calls.append((list(views), kwargs))
        for view in views:
            if view in self.missing:
                raise KeyError(f"View '{view}' not found in any storage location of the dataset.")
        child = _StubDataset()
        child.tag = f"dataset({','.join(views)})"
        child.randomization = kwargs["randomization"]
        return child


def _stub_model(*, samples: list[str] | None = None, drugs: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(config=_StubConfig(samples=samples or [], drugs=drugs or []))


def test_view_test_builders() -> None:
    assert _single_view_tests(["a", "b"], "SVRC") == {("SVRC", "a"): ["a"], ("SVRC", "b"): ["b"]}
    assert _single_view_tests([], "SVRC") == {}
    assert _complement_view_tests(["a", "b", "c"], "SVCC") == {
        ("SVCC", "a"): ["b", "c"],
        ("SVCC", "b"): ["a", "c"],
        ("SVCC", "c"): ["a", "b"],
    }
    assert _complement_view_tests(["a"], "SVCC") == {("SVCC", "a"): []}


@pytest.mark.parametrize(
    ("mode", "model_kwargs", "expected"),
    [
        ("SVRC", {"samples": ["expr", "mut"]}, [["expr"], ["mut"]]),
        ("SVCC", {"samples": ["expr", "mut"]}, [["mut"], ["expr"]]),
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
    assert result[0].mdata.uns["randomization_provenance"]["root_seed"] == 42

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


def test_missing_view_and_empty_complement_are_skipped_with_warnings(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="drevalpy.experiment._randomization"):
        results = randomization(
            _stub_model(samples=["expr"], drugs=["fp"]), _StubDataset(missing=("fp",)), ["SVRC", "SVRD"]
        )
        assert [result.randomization for result in results] == [("SVRC", "expr")]
        assert "Skipping the SVRD randomization test of 'fp'" in caplog.text

        results = randomization(_stub_model(samples=["expr"], drugs=["fp", "smiles"]), _StubDataset(), ["SVCC", "SVCD"])
    assert [result.randomization for result in results] == [("SVCD", "fp"), ("SVCD", "smiles")]
    assert "Skipping the SVCC randomization test of 'expr': no other view to randomize" in caplog.text


def test_real_model_config_views_and_real_dataset(synthetic_dataset, caplog: pytest.LogCaptureFixture) -> None:
    spec = ModelSpec.from_name("ElasticNet")
    stub = _StubDataset()
    randomization(spec, stub, ["SVRC", "SVRD"])
    assert [views for views, _ in stub.calls] == [["gene_expression"], ["morgan_fingerprint"]]

    results = randomization(spec, synthetic_dataset, ["SVRC"], random_state=0)
    assert [ds.randomization for ds in results] == [("SVRC", "gene_expression")]
    assert results[0] is not synthetic_dataset
    assert synthetic_dataset.randomization is None
    provenance = results[0].mdata.uns["randomization_provenance"]
    assert (provenance["root_seed"], provenance["views"]) == (0, ["gene_expression"])

    with caplog.at_level("WARNING", logger="drevalpy.experiment._randomization"):
        bare = synthetic(n_samples=6, n_drugs=2, n_tissues=2)
        assert randomization(spec, bare, ["SVRD", "SVCD", "SVCC"]) == []
    assert "'morgan_fingerprint'" in caplog.text
