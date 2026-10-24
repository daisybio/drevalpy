"""Public ``drevalpy.models.config`` surface; origins are the leaf modules defining each name."""

from __future__ import annotations

from drevalpy.models import config
from tests._barrel_surface import DeclaredSurface

EXPECTED_ORIGINS: dict[str, str] = {
    "SampleFeaturizerConfig": "drevalpy.models.config.featurizer",
    "DrugFeaturizerConfig": "drevalpy.models.config.featurizer",
    "FeaturizerConfig": "drevalpy.models.config.featurizer",
    "ModelConfig": "drevalpy.models.config.model",
    "ModelScope": "drevalpy.types.enums.model_scope",
    "PredictionMode": "drevalpy.types.enums.prediction_mode",
    "PredictorConfig": "drevalpy.models.config.predictor",
    "ResolvedModelConfig": "drevalpy.models.config.resolved",
    "from_dict": "drevalpy.models.config.io",
    "from_spec": "drevalpy.models.config.io",
    "from_yaml": "drevalpy.models.config.io",
    "validate": "drevalpy.models.config.validation",
}


class TestConfigSurface(DeclaredSurface):
    barrel = config
    origins = EXPECTED_ORIGINS
    callable_names = ("from_dict", "from_spec", "from_yaml", "validate")


def test_from_spec_and_validate() -> None:
    cfg = config.from_spec("ElasticNet")
    assert isinstance(cfg, config.ModelConfig)
    assert cfg.predictor.name == "elasticNet"
    config.validate(cfg)
