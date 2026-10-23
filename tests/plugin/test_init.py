"""Tests for the :mod:`drevalpy.plugin` facade.

The facade is a compatibility promise, so the tests here are about the promise
rather than about behaviour: every name in ``__all__`` resolves, every alias
points at the same object as its underlying module (so the facade cannot drift
into holding a stale copy), and the wheel carries the PEP 561 marker that makes
the facade's annotations usable by a plugin's type checker.

The packaging assertions at the bottom live here rather than in a new root-level
guard because they are about the same thing the facade is about - what an
installed consumer can see. ``py.typed`` is what makes the facade's annotations
usable, and the ``src/`` layout is what stops the editable install from also
exporting ``tests``, ``tools`` and ``docs`` into every consumer's ``sys.path``.
"""

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

#: Where each exported name is defined, as ``alias -> module``. Written out
#: rather than derived from ``__module__`` so a symbol silently moving between
#: modules is a test failure and not a silently updated expectation.
EXPECTED_ORIGINS: dict[str, str] = {
    "BlockPredictor": "drevalpy.components.predictors.abstract.block",
    "BlockSpec": "drevalpy.types.data.batch.feature_block",
    "CurveQuality": "drevalpy.data.quality",
    "Dataset": "drevalpy.types.data.dataset",
    "DenseViewDrugFeaturizer": "drevalpy.components.featurizers.drug.base",
    "DenseViewFeaturizer": "drevalpy.components.featurizers._dense_view",
    "DenseViewSampleFeaturizer": "drevalpy.components.featurizers.sample.base",
    "DrugFeatureSource": "drevalpy.types.data.feature_source",
    "DrugFeaturizer": "drevalpy.components.featurizers.drug.base",
    "ExperimentResult": "drevalpy.types.results",
    "FeatureBlock": "drevalpy.types.data.batch.feature_block",
    "FeatureContract": "drevalpy.components.contracts.contracts",
    "FeatureFormat": "drevalpy.components.contracts.contracts",
    "FeatureFreePredictor": "drevalpy.components.predictors.abstract.feature_free",
    "FeatureSource": "drevalpy.types.data.feature_source",
    "Featurizer": "drevalpy.components.featurizers.base",
    "FeaturizerResult": "drevalpy.types.data.batch.feature_block",
    "FeaturizerStorageMixin": "drevalpy.components.featurizers.storage",
    "HPOStrategy": "drevalpy.components.featurizers.base",
    "ImageVisualization": "drevalpy.visualization.base",
    "LiteratureReference": "drevalpy.types.enums.literature_reference",
    "MatrixPredictor": "drevalpy.components.predictors.abstract.matrix",
    "ModelInputBatch": "drevalpy.types.data.batch.model_input_batch",
    "ModelResult": "drevalpy.types.results",
    "ModelScope": "drevalpy.types.enums.model_scope",
    "MuDataLike": "drevalpy.types.data.mudatalike",
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
    "metadata_feature_block": "drevalpy.types.data.batch.feature_block",
    "numeric_feature_block": "drevalpy.types.data.batch.feature_block",
    "observed_mask": "drevalpy.data.splitters._folds",
    "ragged_feature_block": "drevalpy.types.data.batch.feature_block",
    "register_for_sides": "drevalpy.components.featurizers._side_binding",
}

#: The five per-registry ``register_*`` are all spelled ``register`` in their own
#: module, so the shared surface check cannot compare them by name:
#: ``alias -> (module, attribute)``.
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

    def test_the_facade_defines_nothing_itself(self):
        """A definition here would be a second implementation to keep in sync."""
        own = [name for name in plugin.__all__ if getattr(vars(plugin)[name], "__module__", "") == "drevalpy.plugin"]

        assert own == []

    def test_the_five_register_aliases_are_distinct(self):
        registrars = {
            plugin.register_sample_featurizer,
            plugin.register_drug_featurizer,
            plugin.register_predictor,
            plugin.register_splitter,
            plugin.register_visualization,
        }

        assert len(registrars) == 5


class TestPyTypedMarker:
    """PEP 561: without this file, a plugin's type checker treats drevalpy as untyped."""

    def test_the_marker_exists_in_the_source_tree(self):
        assert (REPO_ROOT / "src" / "drevalpy" / "py.typed").is_file()

    @pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to build a wheel")
    def test_the_marker_ships_in_the_built_wheel(self, built_wheel_contents):
        """Built for real: a marker missing from the wheel does nothing for consumers.

        The wheel comes from the session-scoped ``built_wheel_contents`` fixture
        in ``tests/conftest.py``, which is shared with
        ``tests/testing/test_init.py`` so the repository is only built once.
        """
        assert "drevalpy/py.typed" in built_wheel_contents


class TestTheWheelShipsThePackageOnly:
    """Its own repository layout is drevalpy's business, not a consumer's.

    Asserted apart from the ``py.typed`` check above because it is a different
    packaging promise; both read the same session-shared build, so there is no
    longer a reason to bundle them into one test.
    """

    @pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to build a wheel")
    def test_no_repo_only_directory_is_in_the_wheel(self, built_wheel_contents):
        leaked = sorted(name for name in built_wheel_contents if name.startswith(("tests/", "tools/", "docs/")))

        assert leaked == []


class TestEditableInstallDoesNotLeak:
    """A flat layout makes the editable install put the bare project root on ``sys.path``.

    That would make drevalpy's own ``tests``, ``tools`` and ``docs`` importable by
    every consumer of the environment, so a plugin repo's ``import tests.x``
    would resolve to *drevalpy's* tests. With the package under ``src/``, uv_build
    exports only ``src/``.
    """

    def test_the_package_lives_under_src(self, pyproject):
        assert pyproject["build-system"]["build-backend"] == "uv_build"
        assert (REPO_ROOT / "src" / "drevalpy" / "__init__.py").is_file()
        assert not (REPO_ROOT / "drevalpy").exists()

    def test_the_repo_top_level_directories_are_not_importable(self, tmp_path):
        """Run from outside the repo, so only the installed paths are in play."""
        script = (
            "import importlib.util as u\n"
            "print(u.find_spec('drevalpy') is not None)\n"
            "print([name for name in ('tests', 'tools', 'docs') if u.find_spec(name) is not None])\n"
        )

        result = run_trusted_python(script, cwd=str(tmp_path))

        assert result.returncode == 0, result.stderr
        drevalpy_found, leaked = result.stdout.splitlines()[-2:]
        assert drevalpy_found == "True"
        assert leaked == "[]"

    def test_no_pth_file_exports_the_project_root(self):
        """Belt and braces: the mechanism, not just its effect."""
        offenders = [
            path.name
            for site_dir in sys.path
            if site_dir.endswith("site-packages")
            for path in UPath(site_dir).glob("*.pth")
            if str(REPO_ROOT) in path.read_text(encoding="utf-8").splitlines()
        ]

        assert offenders == []


class TestFacadeIsSelfContained:
    #: Extended tier: spawns an interpreter (~2.4s) to prove the facade needs no
    #: other drevalpy import. Sharing a process with another test would defeat it.
    pytestmark = pytest.mark.slow

    def test_importing_only_the_facade_is_enough_to_subclass(self):
        """A plugin importing nothing but the facade must be able to declare a component."""
        script = (
            "from drevalpy.plugin import (\n"
            "    SampleFeaturizer, FeatureFormat, BlockSpec, FeaturizerResult, numeric_feature_block,\n"
            "    register_sample_featurizer,\n"
            ")\n"
            "import numpy as np\n"
            "from typing import ClassVar\n"
            "\n"
            "@register_sample_featurizer(\n"
            "    'facadeProbe', description='probe', contract=FeatureFormat.NUMERIC_MATRIX\n"
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
