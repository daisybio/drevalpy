"""Randomization test utilities for feature importance analysis."""

from __future__ import annotations

import hashlib
import json

from drevalpy.log import get_logger
from drevalpy.models.spec import ModelSpec
from drevalpy.types.data.dataset import Dataset

logger = get_logger(__name__)


def _single_view_tests(views: list[str], prefix: str) -> dict[tuple[str, str], list[str]]:
    """One test per view: randomize that view only."""
    return {(prefix, view): [view] for view in views}


def _complement_view_tests(views: list[str], prefix: str) -> dict[tuple[str, str], list[str]]:
    """One test per view: randomize all views except that one."""
    return {(prefix, view): [v for v in views if v != view] for view in views}


def randomization(
    model_spec: ModelSpec,
    dataset: Dataset,
    randomization_mode: list[str],
    *,
    randomization_type: str = "permutation",
    random_state: int = 42,
) -> list[Dataset]:
    """Generate randomized datasets for feature importance testing.

    For each randomization test (determined by model views and mode), produces
    a copy of the dataset with the relevant views shuffled. Each returned
    dataset has its ``randomization`` field set to ``(mode, view)``. A test that
    would randomize nothing, because the dataset lacks the view or the model has
    no other view to randomize, is skipped with a warning, so no returned dataset
    carries a randomization label it did not earn.

    :param model_spec: Model class whose config defines available views.
    :param dataset: Original dataset to randomize.
    :param randomization_mode: List of mode codes (e.g. ["SVRC", "SVRD"]).
    :param randomization_type: "permutation" or "invariant".
    :param random_state: Seed for reproducibility.
    :returns: List of randomized datasets.
    """
    config = model_spec.config
    sample_views = config.sample_views()
    drug_views = config.drug_views()

    builders = {
        "SVRC": lambda: _single_view_tests(sample_views, "SVRC"),
        "SVCC": lambda: _complement_view_tests(sample_views, "SVCC"),
        "SVRD": lambda: _single_view_tests(drug_views, "SVRD"),
        "SVCD": lambda: _complement_view_tests(drug_views, "SVCD"),
    }

    tests: dict[tuple[str, str], list[str]] = {}
    for mode in randomization_mode:
        if mode in builders:
            tests.update(builders[mode]())

    results: list[Dataset] = []
    for (mode, view), views in tests.items():
        if not views:
            logger.warning("Skipping the %s randomization test of '%s': no other view to randomize", mode, view)
            continue
        views = sorted(views)
        identity = json.dumps([random_state, randomization_type, mode, view, views], separators=(",", ":"))
        seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "big")
        try:
            ds = dataset.with_randomized_views(
                views, randomization_type=randomization_type, random_state=seed, randomization=(mode, view)
            )
        except KeyError as error:
            logger.warning("Skipping the %s randomization test of '%s': %s", mode, view, error.args[0])
            continue
        ds.mdata.uns["randomization_provenance"] = {
            "root_seed": random_state,
            "effective_seed": seed,
            "type": randomization_type,
            "views": views,
        }
        results.append(ds)
    return results
