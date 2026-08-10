"""What a registered predictor implies about the config around it.

A predictor class is written for one training scope, and a single-drug one fits a separate
estimator per drug. Both facts are properties of the component, not choices a ``ModelConfig``
makes, so they are read off the registry rather than stored as fields.

The two readers sit on opposite sides of field validation. ``scope_for_predictor`` takes a
resolved name and is free to raise; ``needs_identity_drug_routing`` runs before validation on
a slot that may be written in any accepted spelling, or be unusable, so it never raises.
"""

from __future__ import annotations

from typing import Any

from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
from drevalpy.components.registry import get_predictor
from drevalpy.models.config.predictor import PredictorConfig
from drevalpy.types.enums.model_scope import ModelScope


def scope(name: str) -> ModelScope:
    """Return the training scope the registered predictor is written for.

    :param name: Registry name of the predictor.
    :returns: The scope declared by the predictor class.
    """
    return get_predictor(name).scope


def needs_identity_drug_routing(slot: Any) -> bool:
    """Report whether a predictor slot names a predictor that routes per drug by identity.

    True for a single-drug predictor that consumes features: it fits one estimator per drug, so
    it needs the drug's identity to dispatch each pair to the right one. Which featurizer that
    is was never a choice, so this answers yes or no rather than naming one.

    Written for a ``mode="before"`` validator, so *slot* may be any spelling a predictor slot
    accepts -- a bare name, a one-key mapping, a ``name`` mapping, or a ``PredictorConfig`` --
    and may equally be unusable. Anything that cannot be read as a registered predictor is
    reported as ``False`` rather than raising, leaving the bad value for normal field validation
    to report. Pydantic's ``ValidationError`` is a ``ValueError``, so one ``ValueError`` catch
    covers both a malformed slot and an unknown predictor name.

    :param slot: Raw value of a config payload's ``predictor`` slot.
    :returns: Whether the slot's predictor needs the identity drug featurizer for routing.
    """
    try:
        cls = get_predictor(PredictorConfig.model_validate(slot).name)
    except (TypeError, ValueError, ImportError):
        return False
    return cls.scope is ModelScope.SINGLE_DRUG and not issubclass(cls, FeatureFreePredictor)
