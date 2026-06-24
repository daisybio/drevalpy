"""Declarative configuration for modular featurizer/predictor pairing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal


class PredictionMode(StrEnum):
    """Whether the model predicts a continuous response or a discrete class."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"


@dataclass
class FeaturizerConfig:
    """Declarative specification for a featurizer."""

    type: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    registry: Literal["cell_line", "drug"] = "cell_line"
    view: str | None = None
    views: list[str] | None = None
    hyperparameter_space: dict[str, dict[str, Any]] | None = None

    def create_instance(self):
        """Instantiate the configured featurizer from the registry."""
        from drevalpy.components.registry import lookup as reg

        if self.registry == "cell_line":
            cls = reg.get_cell_line_featurizer(self.type)
        else:
            cls = reg.get_drug_featurizer(self.type)
        hp = dict(self.hyperparameters)
        if self.view is not None:
            hp.setdefault("view", self.view)
        if self.views is not None:
            hp.setdefault("views", self.views)
        return cls(**hp)


@dataclass
class PredictorConfig:
    """Declarative specification for a predictor."""

    type: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    hyperparameter_space: dict[str, dict[str, Any]] | None = None

    def create_instance(self):
        """Instantiate the configured predictor from the registry."""
        from drevalpy.components.registry import lookup as reg

        cls = reg.get_predictor(self.type)
        return cls()


@dataclass
class ModelConfig:
    """Full declarative specification for a composed model."""

    cell_line_featurizer: FeaturizerConfig | None
    drug_featurizer: FeaturizerConfig | None
    predictor: PredictorConfig
    prediction_mode: PredictionMode = PredictionMode.REGRESSION

    @property
    def model_id(self) -> str | None:
        """Stable identifier for a fully specified combination."""
        if self.cell_line_featurizer is None and self.drug_featurizer is None:
            return self.predictor.type
        if self.cell_line_featurizer is None or self.drug_featurizer is None:
            return None
        return f"{self.cell_line_featurizer.type}:" f"{self.drug_featurizer.type}:" f"{self.predictor.type}"

    def validate(self) -> None:
        """Check registry slots, feature compatibility, and prediction mode."""
        from drevalpy.components.validation import validate_model_config

        validate_model_config(self)

    def create_model(self):
        """Build a runnable :class:`~drevalpy.models.composed_model.ComposedModel`."""
        from drevalpy.models.composed_model import ComposedModel

        self.validate()
        cell_line = self.cell_line_featurizer.create_instance() if self.cell_line_featurizer else None
        drug = self.drug_featurizer.create_instance() if self.drug_featurizer else None
        pred = self.predictor.create_instance()
        return ComposedModel(
            cell_line,
            drug,
            pred,
            predictor_hyperparameters=self.predictor.hyperparameters,
            prediction_mode=self.prediction_mode,
        )

    @classmethod
    def from_spec(
        cls,
        spec: str,
        *,
        hyperparameters: dict[str, Any] | None = None,
        prediction_mode: PredictionMode = PredictionMode.REGRESSION,
    ) -> ModelConfig:
        """Build a config from a recipe, zoo, legacy, or baseline spec string."""
        from drevalpy.models.model_config_spec import build_model_config_from_spec

        return build_model_config_from_spec(
            spec,
            hyperparameters=hyperparameters,
            prediction_mode=prediction_mode,
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> ModelConfig:
        """Load a config from a YAML file."""
        from drevalpy.models.config_io import model_config_from_yaml

        return model_config_from_yaml(path)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        """Build a config from a plain dictionary."""
        from drevalpy.models.config_io import model_config_from_dict

        return model_config_from_dict(data)
