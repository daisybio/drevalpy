"""``drevalpy.testing`` ships in the wheel for plugin authors, so its ``__all__`` is a compatibility promise."""

from __future__ import annotations

import importlib

import pytest

from drevalpy import testing
from tests._barrel_surface import DeclaredSurface
from tests._trusted_subprocess import run_trusted_python

EXPECTED_ORIGINS: dict[str, str] = {
    "ConformanceError": "drevalpy.testing.conformance",
    "ENTRY_POINT_GROUP": "drevalpy.registry._plugins",
    "FEATURIZER_CHECKS": "drevalpy.testing.conformance",
    "PREDICTOR_CHECKS": "drevalpy.testing.conformance",
    "PluginCheckError": "drevalpy.testing.plugins",
    "PluginReport": "drevalpy.testing.plugins",
    "build_synthetic_batch": "drevalpy.testing.batch",
    "check_featurizer_fit_transform": "drevalpy.testing.conformance",
    "check_featurizer_instantiates": "drevalpy.testing.conformance",
    "check_featurizer_state_round_trip": "drevalpy.testing.conformance",
    "check_plugin": "drevalpy.testing.plugins",
    "check_predictor_fit_predict": "drevalpy.testing.conformance",
    "check_predictor_instantiates": "drevalpy.testing.conformance",
    "check_predictor_state_round_trip": "drevalpy.testing.conformance",
    "feature_source_for": "drevalpy.testing.conformance",
    "observed_pairs": "drevalpy.testing.batch",
}

RENAMED_ORIGINS: dict[str, tuple[str, str]] = {
    "build_synthetic_dataset": ("drevalpy.data._synthetic", "synthetic"),
}


class TestTestingSurface(DeclaredSurface):
    barrel = testing
    origins = EXPECTED_ORIGINS
    unpinned_names = tuple(RENAMED_ORIGINS)
    callable_names = tuple(RENAMED_ORIGINS)

    @pytest.mark.parametrize(("alias", "origin"), sorted(RENAMED_ORIGINS.items()))
    def test_renamed_alias_is_the_same_object(self, alias, origin):
        module_name, attribute = origin

        assert getattr(testing, alias) is getattr(importlib.import_module(module_name), attribute)


class TestItShipsInTheWheel:
    pytestmark = pytest.mark.slow

    def test_every_submodule_is_in_the_built_wheel(self, built_wheel_contents):
        expected = {
            f"drevalpy/testing/{name}.py" for name in ("__init__", "batch", "conformance", "plugins", "synthetic")
        }
        assert expected <= built_wheel_contents

    def test_it_is_importable_without_pytest(self):
        script = (
            "import sys\n"
            "sys.modules['pytest'] = None\n"
            "import drevalpy.testing as t\n"
            "t.check_featurizer_instantiates\n"
            "print('ok')\n"
        )
        result = run_trusted_python(script)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().endswith("ok")


def test_the_advertised_workflow_runs_end_to_end():
    from drevalpy.registry import predictor, sample_featurizer

    batch = testing.build_synthetic_batch(testing.build_synthetic_dataset())
    for check in testing.PREDICTOR_CHECKS:
        check(predictor.get("ridge"), batch)
    for check in testing.FEATURIZER_CHECKS:
        check(sample_featurizer.get("identity"))
