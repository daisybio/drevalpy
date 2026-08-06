"""What a registered predictor implies about the config around it.

A predictor class is written for one training scope and, when it trains one estimator per
drug, for one routing drug featurizer. Both are properties of the component, not choices a
``ModelConfig`` makes, so they are read off the registry rather than stored as fields.
"""

from __future__ import annotations

from typing import Any

from drevalpy.components.registry import get_predictor
from drevalpy.models.config.predictor import PredictorConfig
from drevalpy.types.model_scope import ModelScope


def scope_for_predictor(name: str) -> ModelScope:
    """Return the training scope the registered predictor is written for.

    :param name: Registry name of the predictor.
    :returns: The scope declared by the predictor class.
    """
    return get_predictor(name).scope


def routing_drug_featurizer_for_slot(slot: Any) -> str | None:
    """Return the drug featurizer a predictor's per-drug routing needs.

    Written for a ``mode="before"`` validator, so *slot* may be any spelling a predictor slot
    accepts -- a bare name, a one-key mapping, a ``name`` mapping, or a ``PredictorConfig`` --
    and may equally be unusable. Anything that cannot be read as a registered predictor yields
    ``None`` instead of raising, leaving the bad value for normal field validation to report.
    Pydantic's ``ValidationError`` is a ``ValueError``, so one ``ValueError`` catch covers both
    a malformed slot and an unknown predictor name.

    :param slot: Raw value of a config payload's ``predictor`` slot.
    :returns: The predictor's ``routing_drug_featurizer``, or ``None`` when there is none or
        the slot cannot be resolved.
    """
    try:
        cls = get_predictor(PredictorConfig.model_validate(slot).name)
    except (TypeError, ValueError, ImportError):
        return None
    routing = getattr(cls, "routing_drug_featurizer", None)
    return routing if isinstance(routing, str) else None
