"""Tests for randomized-dataset generation used by the feature-importance tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from drevalpy.experiment._randomization import (
    _complement_view_tests,
    _single_view_tests,
    randomization,
)
from drevalpy.models import construct_model


@dataclass
class _StubConfig:
    """Minimal stand-in for the ``ModelConfig`` surface ``randomization`` reads."""

    cell_lines: list[str] = field(default_factory=list)
    drugs: list[str] = field(default_factory=list)

    def cell_line_views(self) -> list[str]:
        return list(self.cell_lines)

    def drug_views(self) -> list[str]:
        return list(self.drugs)


class _StubDataset:
    """Records every ``with_randomized_views`` call and returns a tagged sentinel."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def with_randomized_views(self, views: list[str], **kwargs: Any) -> str:
        self.calls.append((list(views), kwargs))
        return f"dataset({','.join(views)})"


def _stub_model(*, cell_lines: list[str] | None = None, drugs: list[str] | None = None) -> type:
    config = _StubConfig(cell_lines=cell_lines or [], drugs=drugs or [])

    class _StubModel:
        @classmethod
        def model_config(cls) -> _StubConfig:
            return config

    return _StubModel


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
            pytest.param("SVRC", [["expr"], ["mut"]], id="SVRC-single-cell-line-view"),
            pytest.param("SVCC", [["mut"], ["expr"]], id="SVCC-complement-cell-line-view"),
        ],
    )
    def test_cell_line_modes_select_the_right_views(self, mode: str, expected: list[list[str]]) -> None:
        dataset = _StubDataset()

        randomization(_stub_model(cell_lines=["expr", "mut"]), dataset, [mode])

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
        model = _stub_model(cell_lines=["expr", "mut"], drugs=["fp"])

        results = randomization(model, _StubDataset(), ["SVRC", "SVRD"])

        assert results == ["dataset(expr)", "dataset(mut)", "dataset(fp)"]

    def test_tags_each_dataset_with_its_mode_and_view(self) -> None:
        dataset = _StubDataset()

        randomization(_stub_model(cell_lines=["expr", "mut"]), dataset, ["SVRC"])

        assert [kwargs["randomization"] for _, kwargs in dataset.calls] == [
            ("SVRC", "expr"),
            ("SVRC", "mut"),
        ]

    def test_forwards_the_randomization_type_and_seed(self) -> None:
        dataset = _StubDataset()

        randomization(
            _stub_model(cell_lines=["expr"]),
            dataset,
            ["SVRC"],
            randomization_type="invariant",
            random_state=7,
        )

        _, kwargs = dataset.calls[0]
        assert kwargs["randomization_type"] == "invariant"
        assert kwargs["random_state"] == 7

    def test_defaults_to_permutation_without_a_seed(self) -> None:
        dataset = _StubDataset()

        randomization(_stub_model(cell_lines=["expr"]), dataset, ["SVRC"])

        _, kwargs = dataset.calls[0]
        assert kwargs["randomization_type"] == "permutation"
        assert kwargs["random_state"] is None

    def test_unknown_modes_are_ignored(self) -> None:
        dataset = _StubDataset()

        results = randomization(_stub_model(cell_lines=["expr"]), dataset, ["NOPE"])

        assert results == []
        assert dataset.calls == []

    def test_no_modes_yields_no_datasets(self) -> None:
        assert randomization(_stub_model(cell_lines=["expr"]), _StubDataset(), []) == []

    def test_later_modes_override_a_colliding_key(self) -> None:
        dataset = _StubDataset()
        model = _stub_model(cell_lines=["expr", "mut"])

        randomization(model, dataset, ["SVRC", "SVRC"])

        assert len(dataset.calls) == 2

    def test_reads_views_from_a_real_model_config(self) -> None:
        dataset = _StubDataset()

        randomization(construct_model("ElasticNet"), dataset, ["SVRC", "SVRD"])

        assert [views for views, _ in dataset.calls] == [["gene_expression"], ["morgan_fingerprint"]]

    def test_produces_tagged_datasets_from_a_real_dataset(self, synthetic_dataset) -> None:
        results = randomization(construct_model("ElasticNet"), synthetic_dataset, ["SVRC"], random_state=0)

        assert [ds.randomization for ds in results] == [("SVRC", "gene_expression")]
        assert results[0] is not synthetic_dataset
        assert synthetic_dataset.randomization is None
