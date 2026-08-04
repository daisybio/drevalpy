"""Declarative configuration for modular featurizer/predictor pairing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from drevalpy.components.featurizer_config_parse import normalize_featurizer_config
from drevalpy.components.featurizer_label import requires_explicit_view
from drevalpy.components.predictor_config_parse import normalize_predictor_config
from drevalpy.types.model_scope import ModelScope
from drevalpy.types.prediction_mode import PredictionMode


def _infer_scope_for_predictor(pred_cls: type[Any]) -> ModelScope | None:
    supported_scopes = getattr(pred_cls, "supported_scopes", None)
    if supported_scopes is not None and len(supported_scopes) == 1:
        return next(iter(supported_scopes))
    return None


def _normalize_single_drug_identity(data: dict[str, Any]) -> dict[str, Any]:
    """Inject implicit identity drug featurizer for single-drug feature-based configs."""
    from drevalpy.components.registry import get_predictor

    predictor = data.get("predictor")
    if predictor is None:
        return data
    if isinstance(predictor, PredictorConfig):
        predictor_name = predictor.name
    elif isinstance(predictor, dict):
        predictor_name = str(predictor.get("name", ""))
    else:
        predictor_name = str(predictor)
    try:
        pred_cls = get_predictor(predictor_name)
    except (ValueError, ImportError):
        return data

    explicit_scope = "scope" in data
    scope = data.get("scope", ModelScope.MULTI_DRUG)
    if not explicit_scope:
        inferred = _infer_scope_for_predictor(pred_cls)
        if inferred is not None:
            data = {**data, "scope": inferred}
            scope = inferred

    if scope != ModelScope.SINGLE_DRUG:
        return data
    if data.get("cell_line_featurizer") is None:
        return data
    if data.get("drug_featurizer") is not None:
        return data
    if getattr(pred_cls, "routing_drug_featurizer", None) != "identity":
        return data
    return {**data, "drug_featurizer": DrugFeaturizerConfig(name="identity")}


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

        Returns:
            Featurizer instance for this config.
        """
        from drevalpy.components.registry import lookup as reg

        if self.registry == "cell_line":
            cls = reg.get_cell_line_featurizer(self.name)
        else:
            cls = reg.get_drug_featurizer(self.name)
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


class PredictorConfig(BaseModel):
    """Declarative specification for a predictor."""

    model_config = ConfigDict(extra="forbid")

    name: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    hyperparameter_space: dict[str, dict[str, Any]] | None = None

    @field_validator("name", mode="before")
    @classmethod
    def _coerce_name(cls, value: object) -> object:
        return str(value) if value is not None else value

    @model_validator(mode="before")
    @classmethod
    def _coerce_shorthand(cls, data: object) -> object:
        if isinstance(data, str):
            return normalize_predictor_config(data)
        if isinstance(data, dict) and "name" not in data:
            return normalize_predictor_config(data)
        return data

    def create_instance(self):
        """Instantiate the configured predictor from the registry.

        Returns:
            Predictor instance for this config.
        """
        from drevalpy.components.registry import lookup as reg

        cls = reg.get_predictor(self.name)
        return cls(hyperparameters=dict(self.hyperparameters))


def _as_cell_line_featurizer_config(value: object) -> CellLineFeaturizerConfig:
    if isinstance(value, CellLineFeaturizerConfig):
        return value
    if isinstance(value, FeaturizerConfig):
        return CellLineFeaturizerConfig.model_validate(value.model_dump())
    return CellLineFeaturizerConfig.model_validate(value)


def _as_drug_featurizer_config(value: object) -> DrugFeaturizerConfig:
    if isinstance(value, DrugFeaturizerConfig):
        return value
    if isinstance(value, FeaturizerConfig):
        return DrugFeaturizerConfig.model_validate(value.model_dump())
    return DrugFeaturizerConfig.model_validate(value)


class ModelConfig(BaseModel):
    """Full declarative specification for a composed model."""

    model_config = ConfigDict(extra="forbid")

    cell_line_featurizer: CellLineFeaturizerConfig | None = None
    drug_featurizer: DrugFeaturizerConfig | None = None
    predictor: PredictorConfig
    prediction_mode: PredictionMode = PredictionMode.REGRESSION
    scope: ModelScope = ModelScope.MULTI_DRUG

    @model_validator(mode="before")
    @classmethod
    def _normalize_sections(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        cell_line = normalized.get("cell_line_featurizer")
        if cell_line is not None and not isinstance(cell_line, CellLineFeaturizerConfig):
            if isinstance(cell_line, (str, list, dict, FeaturizerConfig)):
                normalized["cell_line_featurizer"] = _as_cell_line_featurizer_config(cell_line)
        drug = normalized.get("drug_featurizer")
        if drug is not None and not isinstance(drug, DrugFeaturizerConfig):
            if isinstance(drug, (str, list, dict, FeaturizerConfig)):
                normalized["drug_featurizer"] = _as_drug_featurizer_config(drug)
        predictor = normalized.get("predictor")
        if predictor is not None and not isinstance(predictor, PredictorConfig):
            if isinstance(predictor, (str, dict)):
                normalized["predictor"] = normalize_predictor_config(predictor)
        return _normalize_single_drug_identity(normalized)

    @property
    def model_id(self) -> str | None:
        """Stable identifier for a fully specified combination."""
        if self.cell_line_featurizer is None and self.drug_featurizer is None:
            return self.predictor.name
        if self.cell_line_featurizer is None:
            return None
        if (
            self.scope == ModelScope.SINGLE_DRUG
            and self.drug_featurizer is not None
            and self.drug_featurizer.name == "identity"
        ):
            return f"{self.cell_line_featurizer.name}:{self.predictor.name}"
        if self.drug_featurizer is None:
            return None
        return f"{self.cell_line_featurizer.name}:" f"{self.drug_featurizer.name}:" f"{self.predictor.name}"

    def validate(self) -> None:  # type: ignore[override]
        """Check registry slots, feature compatibility, and prediction mode.

        Raises:
            ValueError: If featurizers, predictor contracts, or scope are incompatible.
        """
        from drevalpy.models.config_validation import validate_model_config

        normalized = _normalize_single_drug_identity(self.model_dump())
        if normalized != self.model_dump():
            refreshed = ModelConfig.model_validate(normalized)
            self.drug_featurizer = refreshed.drug_featurizer
            self.scope = refreshed.scope
        validate_model_config(self)

    @classmethod
    def from_spec(
        cls,
        spec: str,
        *,
        hyperparameters: dict[str, Any] | None = None,
        prediction_mode: PredictionMode = PredictionMode.REGRESSION,
    ) -> ModelConfig:
        """Build a config from a recipe, zoo, legacy, or baseline spec string.

        Args:
            spec: Zoo preset name, colon-separated recipe, or legacy baseline token.
            hyperparameters: Optional flat public hyperparameter overrides.
            prediction_mode: Regression or classification mode for the predictor.

        Returns:
            Validated ``ModelConfig`` instance.

        Raises:
            ValueError: If *spec* is unknown or validation fails.
        """
        from drevalpy.models.model_config_spec import build_model_config_from_spec

        return build_model_config_from_spec(
            spec,
            hyperparameters=hyperparameters,
            prediction_mode=prediction_mode,
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> ModelConfig:
        """Load a config from a YAML file.

        Args:
            path: Path to a YAML mapping describing the model config.

        Returns:
            Validated ``ModelConfig`` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the YAML content is not a valid config mapping.
        """
        from drevalpy.models.config_io import model_config_from_yaml

        return model_config_from_yaml(path)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        """Build a config from a plain dictionary.

        Args:
            data: Mapping with featurizer and predictor sections.

        Returns:
            Validated ``ModelConfig`` instance.

        Raises:
            ValueError: If validation fails.
        """
        from drevalpy.models.config_io import model_config_from_dict

        return model_config_from_dict(data)
