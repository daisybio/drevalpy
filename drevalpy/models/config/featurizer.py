"""Declarative featurizer configuration schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from drevalpy.components.featurizer_config_parse import normalize_featurizer_config
from drevalpy.components.featurizer_label import requires_explicit_view


class FeaturizerConfig(BaseModel):
    """Declarative specification for a featurizer."""

    model_config = ConfigDict(extra="forbid")

    name: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    registry: Literal["cell_line", "drug"] = "cell_line"
    view: str | None = None
    views: list[str] | None = None
    hyperparameter_space: dict[str, dict[str, Any]] | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_shorthand(cls, data: object) -> object:
        if isinstance(data, (str, list)):
            return normalize_featurizer_config(data)
        if isinstance(data, dict):
            if "name" in data:
                registry = str(data.get("registry", "cell_line"))
                return normalize_featurizer_config(data, default_registry=registry)
            return normalize_featurizer_config(data)
        return data

    @model_validator(mode="after")
    def _require_non_empty_view_fields(self) -> FeaturizerConfig:
        if self.view is not None and not str(self.view).strip():
            msg = "view must be a non-empty string when set"
            raise ValueError(msg)
        if self.views is not None:
            if not self.views:
                msg = "views must be a non-empty list when set"
                raise ValueError(msg)
            if any(not str(view).strip() for view in self.views):
                msg = "views must contain non-empty strings"
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _require_explicit_view_for_parametric_featurizers(self) -> FeaturizerConfig:
        if requires_explicit_view(self.name) and not self.view:
            msg = f"Featurizer {self.name!r} requires an explicit view, e.g. {self.name}[expression]"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _require_unique_qualified_children(self) -> FeaturizerConfig:
        if self.name != "concatFeaturizers":
            return self
        from drevalpy.components.featurizer_tree import (
            ensure_unique_qualified_featurizers,
        )

        ensure_unique_qualified_featurizers(self, str(self.registry))
        return self

    def create_instance(self):
        """Instantiate the configured featurizer from the registry.

        :returns: Featurizer instance for this config.
        """
        from drevalpy.components.registry import get_cell_line_featurizer, get_drug_featurizer

        if self.registry == "cell_line":
            cls = get_cell_line_featurizer(self.name)
        else:
            cls = get_drug_featurizer(self.name)
        hp = dict(self.hyperparameters)
        if self.view is not None:
            hp.setdefault("view", self.view)
        if self.views is not None:
            hp.setdefault("views", self.views)
        return cls(**hp)


class CellLineFeaturizerConfig(FeaturizerConfig):
    """Featurizer config fixed to the cell-line registry."""

    registry: Literal["cell_line"] = "cell_line"

    @model_validator(mode="before")
    @classmethod
    def _coerce_shorthand(cls, data: object) -> object:
        if isinstance(data, (str, list)):
            return normalize_featurizer_config(data, default_registry="cell_line")
        if isinstance(data, dict):
            payload = {**data, "registry": "cell_line"}
            return normalize_featurizer_config(payload, default_registry="cell_line")
        return data


class DrugFeaturizerConfig(FeaturizerConfig):
    """Featurizer config fixed to the drug registry."""

    registry: Literal["drug"] = "drug"

    @model_validator(mode="before")
    @classmethod
    def _coerce_shorthand(cls, data: object) -> object:
        if isinstance(data, (str, list)):
            return normalize_featurizer_config(data, default_registry="drug")
        if isinstance(data, dict):
            payload = {**data, "registry": "drug"}
            return normalize_featurizer_config(payload, default_registry="drug")
        return data
