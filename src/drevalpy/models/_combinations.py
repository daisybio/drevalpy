"""Expansion of component lists into the valid model specs of a grid."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from drevalpy.models.spec import ModelSpec


@dataclass(frozen=True)
class SkippedCombination:
    """A combination of components that was not turned into a model spec.

    Attributes:
        recipe: The recipe that was tried, such as ``"raw[gene_expression]:fingerprints:naiveTissueMean"``.
        reason: Why it was skipped, as reported by config validation.
    """

    recipe: str
    reason: str


@dataclass(frozen=True)
class Combinations:
    """The outcome of :func:`combinations`.

    Attributes:
        specs: One model spec per valid combination, named by its full recipe string.
        skipped: The invalid combinations, each with the reason, so the caller can print them.
    """

    specs: list[ModelSpec]
    skipped: list[SkippedCombination]


def combinations(
    *,
    sample: Sequence[str] = (),
    drug: Sequence[str] = (),
    predictors: Sequence[str],
) -> Combinations:
    """Expand component lists into every valid model, and report the invalid ones.

    Each product ``sample x drug x predictor`` is formatted as the recipe
    ``<sample>:<drug>:<predictor>`` (an empty side is left out), checked with the
    config-time validator, and either turned into a :class:`~drevalpy.models.spec.ModelSpec`
    or recorded as skipped with the validator's reason. Model names are the full recipe
    strings including views, so they are unique; duplicate recipes are dropped.

    Args:
        sample: Sample featurizer recipes, such as ``"pca[gene_expression]"``. Empty means no
            sample featurizer.
        drug: Drug featurizer recipes, such as ``"fingerprints"``. Empty means no drug featurizer.
        predictors: Predictor names, such as ``"elasticNet"``.

    Returns:
        The valid specs and the skipped combinations.

    Raises:
        ValueError: If no combination is valid.
    """
    raise NotImplementedError("W1-P4")
