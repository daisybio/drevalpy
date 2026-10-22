"""Tests for randomized-dataset generation used by the feature-importance tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from drevalpy.experiment._randomization import (
    _complement_view_tests,
    _single_view_tests,
    randomization,
)
from drevalpy.models import ModelSpec


@dataclass
class _StubConfig:
    """Minimal stand-in for the ``ModelConfig`` surface ``randomization`` reads."""

    samples: list[str] = field(default_factory=list)
    drugs: list[str] = field(default_factory=list)

    def sample_views(self) -> list[str]:
        return list(self.samples)

    def drug_views(self) -> list[str]:
        return list(self.drugs)


class _StubDataset:
    """Records randomized children with the Dataset surface the helper writes."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.mdata = SimpleNamespace(uns={})
        self.tag = "dataset"
        self.randomization: tuple[str, str] | None = None

    def with_randomized_views(self, views: list[str], **kwargs: Any) -> _StubDataset:
        self.calls.append((list(views), kwargs))
        child = _StubDataset()
        child.tag = f"dataset({','.join(views)})"
        child.randomization = kwargs["randomization"]
        return child


def _stub_model(*, samples: list[str] | None = None, drugs: list[str] | None = None) -> type:
    config = _StubConfig(samples=samples or [], drugs=drugs or [])

    return SimpleNamespace(config=config)


class TestSingleViewTests:
    def test_randomizes_one_view_per_test(self) -> None:
        assert _single_view_tests(["a", "b"], "SVRC") == {
            ("SVRC", "a"): ["a"],
            ("SVRC", "b"): ["b"],
        }

    def test_no_views_yields_no_tests(self) -> None:
        assert _single_view_tests([], "SVRC") == {}


class TestComplementViewTests:
    def test_randomizes_everything_but_the_named_view(self) -> None:
        assert _complement_view_tests(["a", "b", "c"], "SVCC") == {
            ("SVCC", "a"): ["b", "c"],
            ("SVCC", "b"): ["a", "c"],
            ("SVCC", "c"): ["a", "b"],
        }

    def test_a_single_view_leaves_nothing_to_randomize(self) -> None:
        assert _complement_view_tests(["a"], "SVCC") == {("SVCC", "a"): []}


class TestRandomization:
    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            pytest.param("SVRC", [["expr"], ["mut"]], id="SVRC-single-sample-view"),
            pytest.param("SVCC", [["mut"], ["expr"]], id="SVCC-complement-sample-view"),
        ],
    )
    def test_sample_modes_select_the_right_views(self, mode: str, expected: list[list[str]]) -> None:
        dataset = _StubDataset()

        randomization(_stub_model(samples=["expr", "mut"]), dataset, [mode])

        assert [views for views, _ in dataset.calls] == expected

    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            pytest.param("SVRD", [["fp"], ["smiles"]], id="SVRD-single-drug-view"),
            pytest.param("SVCD", [["smiles"], ["fp"]], id="SVCD-complement-drug-view"),
        ],
    )
    def test_drug_modes_select_the_right_views(self, mode: str, expected: list[list[str]]) -> None:
        dataset = _StubDataset()

        randomization(_stub_model(drugs=["fp", "smiles"]), dataset, [mode])

        assert [views for views, _ in dataset.calls] == expected

    def test_returns_one_dataset_per_test(self) -> None:
        model = _stub_model(samples=["expr", "mut"], drugs=["fp"])

        results = randomization(model, _StubDataset(), ["SVRC", "SVRD"])

        assert [result.tag for result in results] == ["dataset(expr)", "dataset(mut)", "dataset(fp)"]

    def test_tags_each_dataset_with_its_mode_and_view(self) -> None:
        dataset = _StubDataset()

        randomization(_stub_model(samples=["expr", "mut"]), dataset, ["SVRC"])

        assert [kwargs["randomization"] for _, kwargs in dataset.calls] == [
            ("SVRC", "expr"),
            ("SVRC", "mut"),
        ]

    def test_forwards_the_randomization_type_and_seed(self) -> None:
        dataset = _StubDataset()

        randomization(
            _stub_model(samples=["expr"]),
            dataset,
            ["SVRC"],
            randomization_type="invariant",
            random_state=7,
        )

        _, kwargs = dataset.calls[0]
        assert kwargs["randomization_type"] == "invariant"
        assert isinstance(kwargs["random_state"], int)

    def test_defaults_to_permutation_with_the_documented_root_seed(self) -> None:
        dataset = _StubDataset()

        result = randomization(_stub_model(samples=["expr"]), dataset, ["SVRC"])

        _, kwargs = dataset.calls[0]
        assert kwargs["randomization_type"] == "permutation"
        assert isinstance(kwargs["random_state"], int)
        assert result[0].mdata.uns["randomization_provenance"]["root_seed"] == 42

    def test_same_root_seed_is_stable_but_variants_get_distinct_effective_seeds(self) -> None:
        first = _StubDataset()
        second = _StubDataset()
        model = _stub_model(samples=["expr", "mut"])

        randomization(model, first, ["SVRC"], random_state=7)
        randomization(model, second, ["SVRC"], random_state=7)

        first_seeds = [kwargs["random_state"] for _, kwargs in first.calls]
        second_seeds = [kwargs["random_state"] for _, kwargs in second.calls]
        assert first_seeds == second_seeds
        assert len(set(first_seeds)) == len(first_seeds)

    def test_view_order_does_not_change_effective_seeds(self) -> None:
        forward = _StubDataset()
        reverse = _StubDataset()

        randomization(_stub_model(samples=["expr", "mut"]), forward, ["SVRC"], random_state=7)
        randomization(_stub_model(samples=["mut", "expr"]), reverse, ["SVRC"], random_state=7)

        by_view = {kwargs["randomization"][1]: kwargs["random_state"] for _, kwargs in forward.calls}
        reversed_by_view = {kwargs["randomization"][1]: kwargs["random_state"] for _, kwargs in reverse.calls}
        assert by_view == reversed_by_view

    def test_unknown_modes_are_ignored(self) -> None:
        dataset = _StubDataset()

        results = randomization(_stub_model(samples=["expr"]), dataset, ["NOPE"])

        assert results == []
        assert dataset.calls == []

    def test_no_modes_yields_no_datasets(self) -> None:
        assert randomization(_stub_model(samples=["expr"]), _StubDataset(), []) == []

    def test_later_modes_override_a_colliding_key(self) -> None:
        dataset = _StubDataset()
        model = _stub_model(samples=["expr", "mut"])

        randomization(model, dataset, ["SVRC", "SVRC"])

        assert len(dataset.calls) == 2

    def test_reads_views_from_a_real_model_config(self) -> None:
        dataset = _StubDataset()

        randomization(ModelSpec.from_name("ElasticNet"), dataset, ["SVRC", "SVRD"])

        assert [views for views, _ in dataset.calls] == [["gene_expression"], ["morgan_fingerprint"]]

    def test_produces_tagged_datasets_from_a_real_dataset(self, synthetic_dataset) -> None:
        results = randomization(ModelSpec.from_name("ElasticNet"), synthetic_dataset, ["SVRC"], random_state=0)

        assert [ds.randomization for ds in results] == [("SVRC", "gene_expression")]
        assert results[0] is not synthetic_dataset
        assert synthetic_dataset.randomization is None
        assert results[0].mdata.uns["randomization_provenance"]["root_seed"] == 0
        assert results[0].mdata.uns["randomization_provenance"]["views"] == ["gene_expression"]
