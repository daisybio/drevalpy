"""Parse declarative model configs from strings, dicts, and YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from drevalpy.components.config import FeaturizerConfig, ModelConfig, PredictorConfig, PredictionMode
from drevalpy.components.model_id import parse_model_id


def _featurizer_from_dict(data: dict[str, Any], *, registry: str) -> FeaturizerConfig:
    if "type" not in data:
        msg = f"{registry} featurizer config requires 'type'"
        raise ValueError(msg)
    return FeaturizerConfig(
        type=str(data["type"]),
        registry=registry,
        hyperparameters=dict(data.get("hyperparameters", {})),
        view=data.get("view"),
        views=data.get("views"),
        hyperparameter_space=data.get("hyperparameter_space"),
    )


def model_config_from_dict(data: dict[str, Any]) -> ModelConfig:
    """Build a :class:`ModelConfig` from a plain dictionary."""
    if "predictor" not in data:
        msg = "model config requires a 'predictor' section"
        raise ValueError(msg)
    predictor_data = data["predictor"]
    if isinstance(predictor_data, str):
        predictor = PredictorConfig(type=predictor_data)
    elif isinstance(predictor_data, dict):
        if "type" not in predictor_data:
            msg = "predictor config requires 'type'"
            raise ValueError(msg)
        predictor = PredictorConfig(
            type=str(predictor_data["type"]),
            hyperparameters=dict(predictor_data.get("hyperparameters", {})),
            hyperparameter_space=predictor_data.get("hyperparameter_space"),
        )
    else:
        msg = "predictor must be a string or mapping"
        raise TypeError(msg)

    cell_line_featurizer = None
    if data.get("cell_line_featurizer") is not None:
        cell_line_featurizer = _featurizer_from_dict(
            dict(data["cell_line_featurizer"]),
            registry="cell_line",
        )

    drug_featurizer = None
    if data.get("drug_featurizer") is not None:
        drug_featurizer = _featurizer_from_dict(
            dict(data["drug_featurizer"]),
            registry="drug",
        )

    mode = data.get("prediction_mode", PredictionMode.REGRESSION)
    if isinstance(mode, str):
        mode = PredictionMode(mode)

    return ModelConfig(
        cell_line_featurizer=cell_line_featurizer,
        drug_featurizer=drug_featurizer,
        predictor=predictor,
        prediction_mode=mode,
    )


def model_config_from_spec(spec: str) -> ModelConfig:
    """Build a :class:`ModelConfig` from a triple or predictor-only string."""
    cell_line_type, drug_type, predictor_type = parse_model_id(spec.strip())
    if cell_line_type is None:
        return ModelConfig(
            cell_line_featurizer=None,
            drug_featurizer=None,
            predictor=PredictorConfig(type=predictor_type),
        )
    return ModelConfig(
        cell_line_featurizer=FeaturizerConfig(type=cell_line_type, registry="cell_line"),
        drug_featurizer=FeaturizerConfig(type=drug_type, registry="drug"),
        predictor=PredictorConfig(type=predictor_type),
    )


def model_config_from_yaml(path: Path | str) -> ModelConfig:
    """Load a :class:`ModelConfig` from a YAML file."""
    yaml_path = Path(path)
    if not yaml_path.is_file():
        msg = f"Model config YAML not found: {yaml_path}"
        raise FileNotFoundError(msg)
    with yaml_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        msg = f"Model config YAML must contain a mapping: {yaml_path}"
        raise ValueError(msg)
    return model_config_from_dict(data)
