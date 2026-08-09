"""Randomization test utilities for feature importance analysis."""

from __future__ import annotations

from drevalpy.models.drp_model import DRPModel
from drevalpy.types.dataset import Dataset


def _single_view_tests(views: list[str], prefix: str) -> dict[str, list[str]]:
    """One test per view: randomize that view only."""
    return {f"{prefix}_{view}": [view] for view in views}


def _complement_view_tests(views: list[str], prefix: str) -> dict[str, list[str]]:
    """One test per view: randomize all views except that one."""
    return {f"{prefix}_{view}": [v for v in views if v != view] for view in views}


def randomization(
    model_class: type[DRPModel],
    dataset: Dataset,
    randomization_mode: list[str],
    *,
    random_state: int | None = None,
) -> list[Dataset]:
    """Generate randomized datasets for feature importance testing.

    For each randomization test (determined by model views and mode), produces
    a copy of the dataset with the relevant views shuffled. Each returned
    dataset has its ``randomization`` field set to ``(mode_view, views)``.

    :param model_class: Model class whose config defines available views.
    :param dataset: Original dataset to randomize.
    :param randomization_mode: List of mode codes (e.g. ["SVRC", "SVRD"]).
    :param random_state: Seed for reproducibility.
    :returns: List of randomized datasets.
    """
    config = model_class.model_config()
    cell_line_views = config.cell_line_views()
    drug_views = config.drug_views()

    builders = {
        "SVRC": lambda: _single_view_tests(cell_line_views, "SVRC"),
        "SVCC": lambda: _complement_view_tests(cell_line_views, "SVCC"),
        "SVRD": lambda: _single_view_tests(drug_views, "SVRD"),
        "SVCD": lambda: _complement_view_tests(drug_views, "SVCD"),
    }

    tests: dict[str, list[str]] = {}
    for mode in randomization_mode:
        if mode in builders:
            tests.update(builders[mode]())

    results: list[Dataset] = []
    for name, views in tests.items():
        ds = dataset.with_randomized_views(views, random_state=random_state, randomization=(name, views))
        results.append(ds)
    return results
