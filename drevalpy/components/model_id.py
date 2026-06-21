"""Stable model identifier helpers for composed configs."""

from __future__ import annotations

_MODEL_ID_SEP = ":"


def format_model_id(
    cell_line: str | None,
    drug: str | None,
    predictor: str,
) -> str:
    """Build a stable model identifier from component type names."""
    if not predictor:
        msg = "predictor is required"
        raise ValueError(msg)
    if cell_line is None and drug is None:
        return predictor
    if cell_line is None or drug is None:
        msg = "cell_line and drug must both be set or both be None"
        raise ValueError(msg)
    return f"{cell_line}{_MODEL_ID_SEP}{drug}{_MODEL_ID_SEP}{predictor}"


def parse_model_id(model_id: str) -> tuple[str | None, str | None, str]:
    """Parse a model identifier into featurizer and predictor type names."""
    if not model_id or not model_id.strip():
        msg = "model_id must be a non-empty string"
        raise ValueError(msg)
    parts = model_id.split(_MODEL_ID_SEP)
    if len(parts) == 1:
        return None, None, parts[0]
    if len(parts) == 3 and all(part.strip() for part in parts):
        return parts[0], parts[1], parts[2]
    msg = (
        "model_id must be 'predictor' or "
        "'cellLineFeaturizer:drugFeaturizer:predictor'"
    )
    raise ValueError(msg)
