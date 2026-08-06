"""Stable model identifier helpers for composed configs."""

from __future__ import annotations

_MODEL_ID_SEP = ":"


def format_model_id(
    cell_line: str | None,
    drug: str | None,
    predictor: str,
) -> str:
    """Build a stable model identifier from component type names.

    :param cell_line: Cell-line featurizer registry name, or ``None`` for feature-free predictors.
    :param drug: Drug featurizer registry name, or ``None`` when omitted.
    :param predictor: Predictor registry name.

    :returns: Colon-separated model id string.

    :raises ValueError: If *predictor* is empty or component names are inconsistent.
    """
    if not predictor:
        msg = "predictor is required"
        raise ValueError(msg)
    if cell_line is None and drug is None:
        return predictor
    if cell_line is None:
        msg = "cell_line is required when drug is set"
        raise ValueError(msg)
    if drug is None:
        return f"{cell_line}{_MODEL_ID_SEP}{predictor}"
    return f"{cell_line}{_MODEL_ID_SEP}{drug}{_MODEL_ID_SEP}{predictor}"


def parse_model_id(model_id: str) -> tuple[str | None, str | None, str]:
    """Parse a model identifier into featurizer and predictor type names.

    Splitting goes through the shared recipe grammar, so a colon inside a bracketed view
    cannot be mistaken for a slot separator. The featurizer slots are returned as unparsed
    recipe strings, since each is normalized against its own registry.

    :param model_id: ``predictor``, ``cell:predictor``, or ``cell:drug:predictor`` id.

    :returns: ``(cell_line_featurizer, drug_featurizer, predictor)`` names.

    :raises ValueError: If *model_id* is empty or not a recognized format.
    """
    if not model_id or not model_id.strip():
        msg = "model_id must be a non-empty string"
        raise ValueError(msg)
    # Imported lazily: drevalpy.models.__init__ eagerly pulls in models.config, whose
    # spec module imports this one, so a module-scope import here would be circular.
    from drevalpy.models.config._recipe import parse_model_recipe

    return parse_model_recipe(model_id)
