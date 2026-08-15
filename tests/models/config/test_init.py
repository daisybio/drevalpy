"""Tests for the public :mod:`drevalpy.models.config` package surface.

This is the highest fan-in barrel in the package: about sixty-six modules reach
``ModelConfig``, the ``FeaturizerConfig`` pair and the three ``from_*``
constructors through it rather than through the modules that define them. Its
``__all__`` is therefore a compatibility promise, and a rename that forgets the
barrel breaks every one of those dependents at import time.

Only the re-export surface and the two primary constructors are asserted here.
Each config model's validation behaviour is tested beside its defining module -
``test_model.py``, ``test_featurizer.py``, ``test_predictor.py``,
``test_resolved.py``, ``test_io.py`` and ``test_validation.py``.

Origins are recorded against the *leaf* module that defines each name, never
against a sibling barrel: ``ModelScope`` and ``PredictionMode`` are pinned to
``drevalpy.types.enums.*`` rather than to the ``drevalpy.types`` barrel, which
also re-exports them, because comparing one re-export with another cannot fail.
The four assertions driven by that table live in ``tests/_barrel_surface.py``.

The package's private modules - ``_recipe``, ``_featurizer_parse``,
``_predictor_parse``, ``_hp_key_validation`` and friends - are deliberately
absent from the surface, and
:meth:`~tests._barrel_surface.DeclaredSurface.test_all_matches_the_recorded_surface`
is what keeps them absent.
"""

from __future__ import annotations

from drevalpy.models import config
from tests._barrel_surface import DeclaredSurface

#: ``exported name -> module that defines it``.
EXPECTED_ORIGINS: dict[str, str] = {
    "CellLineFeaturizerConfig": "drevalpy.models.config.featurizer",
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
