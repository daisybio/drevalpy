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


def _build_randomization_test_views(
    model_class: type[DRPModel],
    randomization_mode: list[str],
) -> dict[str, list[str]]:
    """Build mapping of test name to views that should be randomized.

    Modes:
        - SVRC: Single View Randomize Cell-line — randomize one CL view at a time
        - SVCC: Single View Complement Cell-line — randomize all CL views except one
        - SVRD: Single View Randomize Drug — randomize one drug view at a time
        - SVCD: Single View Complement Drug — randomize all drug views except one
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

    return tests


def randomization(
    model_class: type[DRPModel],
    dataset: Dataset,
    randomization_mode: list[str],
    *,
    random_state: int | None = None,
) -> list[tuple[str, Dataset]]:
    """Generate randomized datasets for feature importance testing.

    For each randomization test (determined by model views and mode), produces
    a copy of the dataset with the relevant views shuffled.

    :param model_class: Model class whose config defines available views.
    :param dataset: Original dataset to randomize.
    :param randomization_mode: List of mode codes (e.g. ["SVRC", "SVRD"]).
    :param random_state: Seed for reproducibility.
    :returns: List of (test_name, randomized_dataset) tuples.
    """
    tests = _build_randomization_test_views(model_class, randomization_mode)
    return [(name, dataset.with_randomized_views(views, random_state=random_state)) for name, views in tests.items()]
