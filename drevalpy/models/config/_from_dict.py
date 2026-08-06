"""Turn a plain mapping into a validated `~drevalpy.models.config.ModelConfig`.

Separate from ``drevalpy.models.config.io`` so that the spec-string layer can reuse it:
``io`` builds on ``spec``, so ``spec`` cannot import back from ``io``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from drevalpy.models.config.model import ModelConfig


def _format_validation_error(exc: ValidationError, *, source: Path | str | None = None) -> str:
    prefix = "Invalid model config"
    if source is not None:
        prefix = f"{prefix} in {source}"
    details = "; ".join(_format_error_entry(error) for error in exc.errors())
    return f"{prefix}: {details}"


def _format_error_entry(error: Any) -> str:
    location = " -> ".join(str(part) for part in error["loc"])
    if not location:
        return str(error["msg"])
    return f"{location}: {error['msg']}"


def from_dict(data: dict[str, Any], *, source: Path | str | None = None) -> ModelConfig:
    """Build a ``ModelConfig`` from a plain dictionary.

    This is where the registry is consulted: field validation resolves featurizer and
    predictor names, and the model-level validator checks that the combination is legal.

    :param data: Mapping with featurizer and predictor sections.
    :param source: Optional path or label included in validation error messages.
    :returns: Validated ``ModelConfig`` instance.
    :raises ValueError: If validation fails.
    """
    try:
        return ModelConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(_format_validation_error(exc, source=source)) from exc
