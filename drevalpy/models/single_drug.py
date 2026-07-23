"""Reusable marker behavior for models fitted independently per drug."""

from typing import ClassVar


class SingleDrugModelMixin:
    """Mark a model as independently fitted per drug."""

    is_single_drug_model: ClassVar[bool] = True
    requires_drug_featurizer: ClassVar[bool] = False
    drug_views: ClassVar[list[str]] = []
