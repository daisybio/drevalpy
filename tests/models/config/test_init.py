"""Public ``drevalpy.models.config`` surface; origins are the leaf modules defining each name."""

from __future__ import annotations

from drevalpy.models import config
from tests._barrel_surface import DeclaredSurface

EXPECTED_ORIGINS: dict[str, str] = {
    "FeaturizerConfig": "drevalpy.models.config.featurizer",
    "ModelConfig": "drevalpy.models.config.model",
    "PredictorConfig": "drevalpy.models.config.predictor",
    "from_dict": "drevalpy.models.config.io",
    "from_recipe": "drevalpy.models.config.io",
    "from_yaml": "drevalpy.models.config.io",
    "validate": "drevalpy.models.config.validation",
}


class TestConfigSurface(DeclaredSurface):
    barrel = config
    origins = EXPECTED_ORIGINS
    callable_names = ("from_dict", "from_recipe", "from_yaml", "validate")


def test_from_recipe_and_validate() -> None:
    cfg = config.from_recipe("scaledGeneExpression:fingerprints:elasticNet")
    assert isinstance(cfg, config.ModelConfig)
    assert cfg.predictor.name == "elasticNet"
    config.validate(cfg)
