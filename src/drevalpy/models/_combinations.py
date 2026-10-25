"""Expansion of component lists into the valid model specs of a grid."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

from drevalpy.errors import InvalidInputError
from drevalpy.models.config._recipe import format_model_recipe
from drevalpy.models.spec import ModelSpec, UnknownModelError


@dataclass(frozen=True)
class SkippedCombination:
    """A combination of components that was not turned into a model spec.

    :ivar recipe: The recipe that was tried, such as ``"raw[gene_expression]:fingerprints:naiveTissueMean"``.
    :ivar reason: Why it was skipped, as reported by config validation.
    """

    recipe: str
    reason: str


@dataclass(frozen=True)
class Combinations:
    """The outcome of :func:`combinations`.

    :ivar specs: One model spec per valid combination, named by its full recipe string.
    :ivar skipped: The invalid combinations, each with the reason, so the caller can print them.
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
    ``<sample>:<drug>:<predictor>`` (an empty side is left out), built with
    :meth:`~drevalpy.models.spec.ModelSpec.from_name`, and either kept as a spec or recorded
    as skipped with the first line of the reason. Model names are the full recipe strings
    including views, so they are unique; duplicate recipes are dropped.

    :param sample: Sample featurizer recipes, such as ``"pca[gene_expression]"`` or
        ``"raw[gene_expression]+tissue"``. Empty means no sample featurizer.
    :param drug: Drug featurizer recipes, such as ``"fingerprints"``. Empty means no drug
        featurizer; a recipe cannot name a drug featurizer without a sample featurizer.
    :param predictors: Predictor names, such as ``"elasticNet"``.
    :returns: The valid specs and the skipped combinations.
    :raises InvalidInputError: If *predictors* is empty, *drug* is given without *sample*, an
        entry is not a plain recipe token, or no combination is valid.
    """
    sample_tokens = _tokens("sample", sample)
    drug_tokens = _tokens("drug", drug)
    predictor_tokens = _tokens("predictors", predictors)
    if not predictor_tokens:
        raise InvalidInputError("combinations needs at least one predictor")
    if drug_tokens and not sample_tokens:
        raise InvalidInputError(
            "drug featurizers need a sample featurizer: a recipe cannot name a drug slot without a sample slot"
        )

    specs: list[ModelSpec] = []
    skipped: list[SkippedCombination] = []
    for sample_token, drug_token, predictor in product(
        sample_tokens or (None,), drug_tokens or (None,), predictor_tokens
    ):
        recipe = format_model_recipe(sample_token, drug_token, predictor)
        try:
            specs.append(ModelSpec.from_name(recipe))
        except UnknownModelError as exc:
            skipped.append(SkippedCombination(recipe, exc.detail.splitlines()[0]))
        except ValueError as exc:
            skipped.append(SkippedCombination(recipe, str(exc).splitlines()[0]))
    if not specs:
        reasons = "\n".join(f"  {item.recipe}: {item.reason}" for item in skipped)
        raise InvalidInputError(f"No valid combination among {len(skipped)}:\n{reasons}")
    return Combinations(specs=specs, skipped=skipped)


def _tokens(label: str, values: Sequence[str]) -> list[str]:
    """Check and de-duplicate one axis of the grid, keeping the given order."""
    if isinstance(values, str):
        raise InvalidInputError(f"{label} must be a list of recipe tokens, not the string {values!r}")
    for value in values:
        if not isinstance(value, str) or not value.strip() or ":" in value:
            raise InvalidInputError(f"{label} entries must be non-empty recipe tokens without ':', got {value!r}")
    return list(dict.fromkeys(value.strip() for value in values))
