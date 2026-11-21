"""The ``drevalpy.plugin`` facade is a compatibility promise; also packaging checks on what consumers see."""

from __future__ import annotations

import importlib
import sys
import textwrap
import tomllib

import pytest
from upath import UPath

from drevalpy import plugin
from tests._barrel_surface import DeclaredSurface
from tests._trusted_subprocess import run_trusted_python

REPO_ROOT = UPath(__file__).resolve().parents[2]

EXPECTED_ORIGINS: dict[str, str] = {
    "Accelerator": "drevalpy.types.enums.accelerator",
    "BlockSpec": "drevalpy.types.data.batch.feature_block",
    "CurveQuality": "drevalpy.data.quality",
    "Dataset": "drevalpy.types.data.dataset._dataset",
    "DrEvalError": "drevalpy.errors",
    "ExperimentResult": "drevalpy.types.results.experiment",
    "FeatureBlock": "drevalpy.types.data.batch.feature_block",
    "FeatureFormat": "drevalpy.components.contracts.feature_format",
    "FeatureFreePredictor": "drevalpy.components.predictors.abstract.feature_free",
    "FeatureMismatchError": "drevalpy.components.featurizers._dense",
    "PrecomputableFeaturizer": "drevalpy.components.featurizers._dense",
    "FeatureSource": "drevalpy.types.data.feature_source",
    "Featurizer": "drevalpy.components.featurizers.featurizer",
    "FeaturizerResult": "drevalpy.types.data.batch.feature_block",
    "Inputs": "drevalpy.components.predictors.inputs",
    "InvalidInputError": "drevalpy.errors",
    "LiteratureReference": "drevalpy.registry._component",
    "MatrixPredictor": "drevalpy.components.predictors.abstract.matrix",
    "ModelInputBatch": "drevalpy.types.data.batch.model_input_batch",
    "ModelResult": "drevalpy.types.results.model",
    "ModelScope": "drevalpy.types.enums.model_scope",
    "Plot": "drevalpy.visualization.base",
    "PlotRequirement": "drevalpy.visualization.requirements",
    "PredictionMode": "drevalpy.types.enums.prediction_mode",
    "Predictor": "drevalpy.components.predictors.abstract.base",
    "Section": "drevalpy.visualization.base",
    "Side": "drevalpy.types.data.feature_source",
    "SingleDrugPredictor": "drevalpy.components.predictors.abstract.single_drug",
    "PairMask": "drevalpy.types.data.pair_mask",
    "Fold": "drevalpy.types.data.fold",
    "SplitValidationError": "drevalpy.registry.splitter._validation",
    "Splitter": "drevalpy.registry.splitter._registry",
    "UnknownNameError": "drevalpy.errors",
    "Validation": "drevalpy.registry.splitter._validation",
    "all_nan_rows_mask": "drevalpy.components.featurizers.featurizer",
    "curve_quality_mask": "drevalpy.data.quality",
    "dense_result": "drevalpy.components.featurizers._dense",
    "global_feature_block": "drevalpy.types.data.batch.feature_block",
    "load_extensions": "drevalpy.registry._extensions",
    "observed_mask": "drevalpy.data.splitters._folds",
    "save_png": "drevalpy.visualization.base",
    "show": "drevalpy.visualization.base",
    "view_matrix": "drevalpy.components.featurizers._dense",
}

RENAMED_ORIGINS: dict[str, tuple[str, str]] = {
    "register_featurizer": ("drevalpy.registry.featurizer._base", "register"),
    "register_predictor": ("drevalpy.registry.predictor", "register"),
    "register_splitter": ("drevalpy.registry.splitter._registry", "register"),
    "register_visualization": ("drevalpy.registry.visualization", "register"),
}


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


class TestPluginSurface(DeclaredSurface):
    barrel = plugin
    origins = EXPECTED_ORIGINS
    unpinned_names = tuple(RENAMED_ORIGINS)

    @pytest.mark.parametrize(("alias", "origin"), sorted(RENAMED_ORIGINS.items()))
    def test_renamed_alias_is_the_same_object(self, alias, origin):
        module_name, attribute = origin

        assert getattr(plugin, alias) is getattr(importlib.import_module(module_name), attribute)

    def test_nothing_public_is_left_out_of_all(self):
        public = {name for name in vars(plugin) if not name.startswith("_")}
        modules = {name for name in public if isinstance(vars(plugin)[name], type(importlib))}

        assert public - modules == set(plugin.__all__)

    def test_the_facade_defines_nothing_itself_and_the_register_aliases_are_distinct(self):
        own = [name for name in plugin.__all__ if getattr(vars(plugin)[name], "__module__", "") == "drevalpy.plugin"]
        assert own == []
        assert len({getattr(plugin, alias) for alias in RENAMED_ORIGINS}) == len(RENAMED_ORIGINS)


def test_the_wheel_ships_py_typed_and_the_package_only(built_wheel_contents):
    assert (REPO_ROOT / "src" / "drevalpy" / "py.typed").is_file()
    assert "drevalpy/py.typed" in built_wheel_contents
    leaked = sorted(name for name in built_wheel_contents if name.startswith(("tests/", "tools/", "docs/")))
    assert leaked == []


class TestEditableInstallDoesNotLeak:
    def test_the_package_lives_under_src(self, pyproject):
        assert pyproject["build-system"]["build-backend"] == "uv_build"
        assert (REPO_ROOT / "src" / "drevalpy" / "__init__.py").is_file()
        assert not (REPO_ROOT / "drevalpy").exists()

    def test_the_repo_top_level_directories_are_not_importable(self, tmp_path):
        # Only the repo's own directories count: a third-party wheel may ship a top-level ``docs`` of its own.
        script = (
            "import importlib.util as u\n"
            "def paths(spec):\n"
            "    return [spec.origin or '', *(spec.submodule_search_locations or [])]\n"
            "print(u.find_spec('drevalpy') is not None)\n"
            f"root = {str(REPO_ROOT)!r}\n"
            "print([name for name in ('tests', 'tools', 'docs')\n"
            "       if (spec := u.find_spec(name)) is not None\n"
            "       and any(p.startswith(root) and 'site-packages' not in p for p in paths(spec))])\n"
        )

        result = run_trusted_python(script, cwd=str(tmp_path))

        assert result.returncode == 0, result.stderr
        drevalpy_found, leaked = result.stdout.splitlines()[-2:]
        assert drevalpy_found == "True"
        assert leaked == "[]"

    def test_no_pth_file_exports_the_project_root(self):
        offenders = [
            path.name
            for site_dir in sys.path
            if site_dir.endswith("site-packages")
            for path in UPath(site_dir).glob("*.pth")
            if str(REPO_ROOT) in path.read_text(encoding="utf-8").splitlines()
        ]

        assert offenders == []


class TestFacadeIsSelfContained:
    pytestmark = pytest.mark.slow

    def test_importing_only_the_facade_is_enough_to_subclass(self):
        script = textwrap.dedent(
            """
            import numpy as np

            from drevalpy.plugin import Featurizer, dense_result, register_featurizer

            @register_featurizer("facadeProbe", side="sample", description="probe")
            class Probe(Featurizer):
                '''Probe.'''

                input_views = ("probe",)

                def fit(self, source, train_ids, *, train_pair_ids=None, es_pair_ids=None):
                    return self

                def transform(self, source, ids):
                    return dense_result(ids, "probe", np.zeros((len(ids), 1)), None)

            from drevalpy.registry.featurizer import names_for_side
            print("facadeProbe" in names_for_side("sample"))
            """
        )

        result = run_trusted_python(script)

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().endswith("True")
