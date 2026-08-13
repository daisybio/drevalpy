"""Tests for the :mod:`drevalpy.testing` package surface.

``drevalpy.testing`` is shipped in the wheel so third-party plugins can import
it, which makes its ``__all__`` a compatibility promise in the same way
:mod:`drevalpy.plugin`'s is. The re-exports are pinned here accordingly.
"""

from __future__ import annotations

import importlib
import shutil
import zipfile

import pytest
from upath import UPath

from drevalpy import testing
from tests._trusted_subprocess import run_trusted_python

REPO_ROOT = UPath(__file__).resolve().parents[2]

#: ``alias -> (submodule, attribute)`` for every re-export.
EXPECTED_ORIGINS: dict[str, tuple[str, str]] = {
    "ENTRY_POINT_GROUP": ("drevalpy.testing.plugins", "ENTRY_POINT_GROUP"),
    "FEATURIZER_CHECKS": ("drevalpy.testing.conformance", "FEATURIZER_CHECKS"),
    "PREDICTOR_CHECKS": ("drevalpy.testing.conformance", "PREDICTOR_CHECKS"),
    "ConformanceError": ("drevalpy.testing.conformance", "ConformanceError"),
    "PluginCheckError": ("drevalpy.testing.plugins", "PluginCheckError"),
    "PluginReport": ("drevalpy.testing.plugins", "PluginReport"),
    "build_synthetic_batch": ("drevalpy.testing.batch", "build_synthetic_batch"),
    "build_synthetic_dataset": ("drevalpy.testing.synthetic", "build_synthetic_dataset"),
    "check_featurizer_fit_transform": ("drevalpy.testing.conformance", "check_featurizer_fit_transform"),
    "check_featurizer_instantiates": ("drevalpy.testing.conformance", "check_featurizer_instantiates"),
    "check_featurizer_state_round_trip": ("drevalpy.testing.conformance", "check_featurizer_state_round_trip"),
    "check_plugin": ("drevalpy.testing.plugins", "check_plugin"),
    "check_predictor_fit_predict": ("drevalpy.testing.conformance", "check_predictor_fit_predict"),
    "check_predictor_instantiates": ("drevalpy.testing.conformance", "check_predictor_instantiates"),
    "check_predictor_state_round_trip": ("drevalpy.testing.conformance", "check_predictor_state_round_trip"),
    "feature_source_for": ("drevalpy.testing.conformance", "feature_source_for"),
    "observed_pairs": ("drevalpy.testing.batch", "observed_pairs"),
}


class TestPublicSurface:
    @pytest.mark.parametrize("name", sorted(testing.__all__))
    def test_every_exported_name_resolves(self, name):
        assert hasattr(testing, name)

    def test_all_is_sorted_and_unique(self):
        assert list(testing.__all__) == sorted(set(testing.__all__))

    def test_every_export_has_a_recorded_origin(self):
        assert sorted(EXPECTED_ORIGINS) == sorted(testing.__all__)

    @pytest.mark.parametrize(("alias", "origin"), sorted(EXPECTED_ORIGINS.items()))
    def test_alias_is_the_same_object(self, alias, origin):
        module_name, attribute = origin
        module = importlib.import_module(module_name)

        assert getattr(testing, alias) is getattr(module, attribute)


class TestItShipsInTheWheel:
    """The whole point of ``drevalpy.testing`` over drevalpy's own ``tests/``.

    A plugin author cannot import an unshipped module, which is why every
    consumer so far had to hand-roll a synthetic dataset.
    """

    @pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to build a wheel")
    def test_every_submodule_is_in_the_built_wheel(self, tmp_path):
        result = run_trusted_python(
            "import subprocess, sys; "
            "sys.exit(subprocess.run(['uv', 'build', '--wheel', '--out-dir', sys.argv[1]]).returncode)",
            cwd=str(REPO_ROOT),
            extra_args=[str(tmp_path)],
        )
        assert result.returncode == 0, result.stderr

        wheels = list(tmp_path.glob("*.whl"))
        assert wheels, "no wheel was produced"
        with zipfile.ZipFile(wheels[0]) as archive:
            shipped = set(archive.namelist())

        expected = {
            "drevalpy/testing/__init__.py",
            "drevalpy/testing/batch.py",
            "drevalpy/testing/conformance.py",
            "drevalpy/testing/plugins.py",
            "drevalpy/testing/synthetic.py",
        }
        assert expected <= shipped

    def test_it_is_importable_without_pytest(self):
        """A plugin may run the checks from a plain script, so nothing here needs pytest."""
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


class TestEndToEndUsage:
    """The kit's own advertised workflow, in the order a plugin author meets it."""

    def test_a_dataset_and_batch_compose_into_a_trained_predictor(self):
        from drevalpy.registry import predictor

        dataset = testing.build_synthetic_dataset()
        batch = testing.build_synthetic_batch(dataset)

        for check in testing.PREDICTOR_CHECKS:
            check(predictor.get("ridge"), batch)

    def test_a_featurizer_can_be_checked_with_no_arguments_at_all(self):
        from drevalpy.registry import cell_line_featurizer

        for check in testing.FEATURIZER_CHECKS:
            check(cell_line_featurizer.get("identity"))
