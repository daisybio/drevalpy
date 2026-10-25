"""The ``drevalpy.plugin`` facade is a compatibility promise; also packaging checks on what consumers see."""

from __future__ import annotations

import importlib
import shutil
import sys
import tomllib

import pytest
from upath import UPath

from drevalpy import plugin
from tests._barrel_surface import DeclaredSurface
from tests._trusted_subprocess import run_trusted_python

REPO_ROOT = UPath(__file__).resolve().parents[2]

EXPECTED_ORIGINS: dict[str, str] = {
    "BlockPredictor": "drevalpy.components.predictors.abstract.block",
    "BlockSpec": "drevalpy.types.data.batch.feature_block",
    "CurveQuality": "drevalpy.data.quality",
    "Dataset": "drevalpy.types.data.dataset._dataset",
    "DenseViewDrugFeaturizer": "drevalpy.components.featurizers.drug.base",
    "DenseViewFeaturizer": "drevalpy.components.featurizers._dense_view",
    "DenseViewSampleFeaturizer": "drevalpy.components.featurizers.sample.base",
    "DrugFeatureSource": "drevalpy.types.data.feature_source",
    "DrugFeaturizer": "drevalpy.components.featurizers.drug.base",
    "ExperimentResult": "drevalpy.types.results",
    "FeatureBlock": "drevalpy.types.data.batch.feature_block",
    "FeatureFormat": "drevalpy.components.contracts.feature_format",
    "FeatureFreePredictor": "drevalpy.components.predictors.abstract.feature_free",
    "FeatureSource": "drevalpy.types.data.feature_source",
    "Featurizer": "drevalpy.components.featurizers.base",
    "FeaturizerResult": "drevalpy.types.data.batch.feature_block",
    "FeaturizerStorageMixin": "drevalpy.components.featurizers.storage",
    "HPOStrategy": "drevalpy.components.featurizers.base",
    "ImageVisualization": "drevalpy.visualization.base",
    "LiteratureReference": "drevalpy.registry.components._metadata",
    "MatrixPredictor": "drevalpy.components.predictors.abstract.matrix",
    "ModelInputBatch": "drevalpy.types.data.batch.model_input_batch",
    "ModelResult": "drevalpy.types.results",
    "ModelScope": "drevalpy.types.enums.model_scope",
    "PlotRequirement": "drevalpy.visualization.requirements",
    "PredictionMode": "drevalpy.types.enums.prediction_mode",
    "Predictor": "drevalpy.components.predictors.abstract.base",
    "ResponseBatch": "drevalpy.types.data.batch.response_batch",
    "RunResult": "drevalpy.types.results",
    "SampleFeatureSource": "drevalpy.types.data.feature_source",
    "SampleFeaturizer": "drevalpy.components.featurizers.sample.base",
    "Section": "drevalpy.visualization.base",
    "SplitMask": "drevalpy.types.data.split_mask",
    "SplitMasks": "drevalpy.types.data.split_masks",
    "SplitValidationError": "drevalpy.registry.splitter",
    "Splitter": "drevalpy.registry.splitter",
    "TrainingContext": "drevalpy.components.contracts.training_context",
    "Validation": "drevalpy.registry.splitter",
    "Visualization": "drevalpy.visualization.base",
    "curve_quality_mask": "drevalpy.data.quality",
    "graph_feature_block": "drevalpy.types.data.batch.feature_block",
    "merge_feature_blocks": "drevalpy.types.data.batch.feature_block",
    "global_feature_block": "drevalpy.types.data.batch.feature_block",
    "numeric_feature_block": "drevalpy.types.data.batch.feature_block",
    "observed_mask": "drevalpy.data.splitters._folds",
    "ragged_feature_block": "drevalpy.types.data.batch.feature_block",
    "register_for_sides": "drevalpy.components.featurizers._side_binding",
}

RENAMED_ORIGINS: dict[str, tuple[str, str]] = {
    "register_sample_featurizer": ("drevalpy.registry.sample_featurizer", "register"),
    "register_drug_featurizer": ("drevalpy.registry.drug_featurizer", "register"),
    "register_predictor": ("drevalpy.registry.predictor", "register"),
    "register_splitter": ("drevalpy.registry.splitter", "register"),
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


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to build a wheel")
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
        script = (
            "from drevalpy.plugin import (\n"
            "    SampleFeaturizer, FeatureFormat, BlockSpec, FeaturizerResult, numeric_feature_block,\n"
            "    register_sample_featurizer,\n"
            ")\n"
            "import numpy as np\n"
            "from typing import ClassVar\n"
            "\n"
            "@register_sample_featurizer(\n"
            "    'facadeProbe', description='probe', output_format=FeatureFormat.NUMERIC_MATRIX\n"
            ")\n"
            "class Probe(SampleFeaturizer):\n"
            "    '''Probe.'''\n"
            "    entity_id_only: ClassVar[bool] = True\n"
            "    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (\n"
            "        BlockSpec('probe', FeatureFormat.NUMERIC_MATRIX),\n"
            "    )\n"
            "    def _fit(self, source, **kwargs):\n"
            "        return self\n"
            "    def _transform_features(self, source, entity_ids):\n"
            "        values = np.zeros((len(entity_ids), 1), dtype=np.float32)\n"
            "        return FeaturizerResult(values, {'probe': numeric_feature_block(values)})\n"
            "    @property\n"
            "    def output_dim(self):\n"
            "        return 1\n"
            "\n"
            "from drevalpy.registry import sample_featurizer\n"
            "print('facadeProbe' in sample_featurizer.list())\n"
        )

        result = run_trusted_python(script)

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().endswith("True")
