"""Declarative featurizer configuration schemas."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator, model_validator

from drevalpy.components.featurizer_config_parse import normalize_featurizer_config
from drevalpy.components.featurizer_label import requires_explicit_view
from drevalpy.models.config.immutable import FrozenMapping, thaw_value


class FeaturizerConfig(BaseModel):
    """Immutable template for a featurizer node in a model stack."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    registry: Literal["cell_line", "drug"] = "cell_line"
    view: str | None = None
    views: tuple[str, ...] | None = None
    featurizers: tuple[FeaturizerConfig, ...] | None = None
    options: FrozenMapping | None = None
    hyperparameter_space: FrozenMapping | None = None

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

    @field_validator("views", mode="before")
    @classmethod
    def _coerce_views(cls, value: object) -> object:
        if value is None or isinstance(value, tuple):
            return value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return tuple(value)
        return value

    @field_validator("featurizers", mode="before")
    @classmethod
    def _coerce_featurizers(cls, value: object) -> object:
        if value is None or isinstance(value, tuple):
            return value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return tuple(value)
        return value

    @field_serializer("views", when_used="always")
    def _serialize_views(self, value: tuple[str, ...] | None) -> list[str] | None:
        return None if value is None else list(value)

    @field_serializer("featurizers", when_used="always")
    def _serialize_featurizers(self, value: tuple[FeaturizerConfig, ...] | None) -> list[dict[str, Any]] | None:
        if value is None:
            return None
        return [child.model_dump(mode="python") for child in value]

    @model_validator(mode="after")
    def _validate_hyperparameter_space(self) -> FeaturizerConfig:
        if self.hyperparameter_space is not None:
            from drevalpy.components.hyperparameter_space import validate_hyperparameter_space

            validate_hyperparameter_space(
                self.hyperparameter_space,
                context=f"FeaturizerConfig({self.name!r}).hyperparameter_space",
            )
        return self

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
    def _require_concat_children(self) -> FeaturizerConfig:
        if self.name == "concatFeaturizers":
            if not self.featurizers:
                msg = "concatFeaturizers requires a non-empty featurizers list"
                raise ValueError(msg)
        elif self.featurizers is not None:
            msg = f"Featurizer {self.name!r} does not accept nested featurizers"
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

    def create_instance(self, hyperparameters: Mapping[str, Any] | None = None):
        """Instantiate the configured featurizer from the registry.

        :param hyperparameters: Concrete constructor values for this node. Nested
            concat children should already be resolved by the caller into instances
            or config payloads under the ``featurizers`` key.
        :returns: Featurizer instance for this config.
        """
        from drevalpy.components.registry import get_cell_line_featurizer, get_drug_featurizer

        if self.registry == "cell_line":
            cls = get_cell_line_featurizer(self.name)
        else:
            cls = get_drug_featurizer(self.name)
        hp = thaw_value(dict(self.options or {}))
        hp.update(thaw_value(dict(hyperparameters or {})))
        if self.view is not None:
            hp.setdefault("view", self.view)
        if self.views is not None:
            hp.setdefault("views", list(self.views))
        if self.featurizers is not None and "featurizers" not in hp:
            hp["featurizers"] = [child.model_dump(mode="python") for child in self.featurizers]
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


FeaturizerConfig.model_rebuild()
CellLineFeaturizerConfig.model_rebuild()
DrugFeaturizerConfig.model_rebuild()
