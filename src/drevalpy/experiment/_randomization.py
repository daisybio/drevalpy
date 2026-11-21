"""Randomization test utilities for feature importance analysis."""

from __future__ import annotations

import hashlib
import json

from drevalpy.log import get_logger
from drevalpy.models.spec import ModelSpec
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import Side
from drevalpy.types.enums import RandomizationType

logger = get_logger(__name__)


#: Mode code -> (the views it ranges over, whether each test randomizes that view alone or all the others).
_MODES: dict[str, tuple[Side, str]] = {
    "SVRC": (Side.SAMPLE, "single"),
    "SVCC": (Side.SAMPLE, "complement"),
    "SVRD": (Side.DRUG, "single"),
    "SVCD": (Side.DRUG, "complement"),
}


def _planned_tests(model_spec: ModelSpec, modes: list[str]) -> dict[tuple[str, str], list[str]]:
    """Return the views each ``(mode, view)`` randomization test replaces; unknown mode codes are ignored.

    :param model_spec: Model whose featurizers define the available views.
    :param modes: Mode codes.
    :returns: Views to randomize, by ``(mode, view)``.
    """
    views_of = {side: root.views() for side, root in model_spec.config.featurizers().items()}
    tests: dict[tuple[str, str], list[str]] = {}
    for mode in modes:
        if mode not in _MODES:
            continue
        side, scope = _MODES[mode]
        for view in views_of.get(side, []):
            tests[(mode, view)] = [view] if scope == "single" else [v for v in views_of[side] if v != view]
    return tests


def randomization(
    model_spec: ModelSpec,
    dataset: Dataset,
    randomization_mode: list[str],
    *,
    randomization_type: RandomizationType = RandomizationType.PERMUTATION,
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
    :param randomization_type: How to replace the view.
    :param random_state: Seed for reproducibility.
    :returns: List of randomized datasets.
    """
    tests = _planned_tests(model_spec, randomization_mode)

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
        ds.randomization_provenance = {
            "root_seed": random_state,
            "effective_seed": seed,
            "type": RandomizationType(randomization_type).value,
            "views": views,
        }
        results.append(ds)
    return results
