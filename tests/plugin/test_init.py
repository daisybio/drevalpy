"""Tests for the :mod:`drevalpy.plugin` facade.

The facade is a compatibility promise, so the tests here are about the promise
rather than about behaviour: every name in ``__all__`` resolves, every alias
points at the same object as its underlying module (so the facade cannot drift
into holding a stale copy), and the wheel carries the PEP 561 marker that makes
the facade's annotations usable by a plugin's type checker.

The packaging assertions at the bottom live here rather than in a new root-level
guard because they are about the same thing the facade is about - what an
installed consumer can see. ``py.typed`` is what makes the facade's annotations
usable, and ``dev-mode-exact`` is what stops the editable install from also
exporting ``tests``, ``tools`` and ``docs`` into every consumer's ``sys.path``.
"""

from __future__ import annotations

import importlib
import shutil
import sys
import tomllib
import zipfile

import pytest
from upath import UPath

from drevalpy import plugin
from tests._trusted_subprocess import run_trusted_python

REPO_ROOT = UPath(__file__).resolve().parents[2]

#: Where each exported name is defined, as ``alias -> (module, attribute)``.
#: Written out rather than derived from ``__module__`` so a symbol silently
#: moving between modules is a test failure and not a silently updated
#: expectation. ``register_*`` are all spelled ``register`` in their own module.
EXPECTED_ORIGINS: dict[str, tuple[str, str]] = {
    "BlockPredictor": ("drevalpy.components.predictors.abstract.block", "BlockPredictor"),
    "BlockSpec": ("drevalpy.types.data.batch.feature_block", "BlockSpec"),
    "CellLineFeatureSource": ("drevalpy.types.data.feature_source", "CellLineFeatureSource"),
    "CellLineFeaturizer": ("drevalpy.components.featurizers.cell_line.base", "CellLineFeaturizer"),
    "Dataset": ("drevalpy.types.data.dataset", "Dataset"),
    "DrugFeatureSource": ("drevalpy.types.data.feature_source", "DrugFeatureSource"),
    "DrugFeaturizer": ("drevalpy.components.featurizers.drug.base", "DrugFeaturizer"),
    "ExperimentResult": ("drevalpy.types.results", "ExperimentResult"),
    "FeatureBlock": ("drevalpy.types.data.batch.feature_block", "FeatureBlock"),
    "FeatureContract": ("drevalpy.components.contracts.contracts", "FeatureContract"),
    "FeatureFormat": ("drevalpy.components.contracts.contracts", "FeatureFormat"),
    "FeatureFreePredictor": ("drevalpy.components.predictors.abstract.feature_free", "FeatureFreePredictor"),
    "FeatureSource": ("drevalpy.types.data.feature_source", "FeatureSource"),
    "Featurizer": ("drevalpy.components.featurizers.base", "Featurizer"),
    "HPOStrategy": ("drevalpy.components.featurizers.base", "HPOStrategy"),
    "ImageVisualization": ("drevalpy.visualization.base", "ImageVisualization"),
    "LiteratureReference": ("drevalpy.types.enums.literature_reference", "LiteratureReference"),
    "MatrixPredictor": ("drevalpy.components.predictors.abstract.matrix", "MatrixPredictor"),
    "ModelInputBatch": ("drevalpy.types.data.batch.model_input_batch", "ModelInputBatch"),
    "ModelResult": ("drevalpy.types.results", "ModelResult"),
    "ModelScope": ("drevalpy.types.enums.model_scope", "ModelScope"),
    "MuDataLike": ("drevalpy.types.data.mudatalike", "MuDataLike"),
    "PlotRequirement": ("drevalpy.visualization.requirements", "PlotRequirement"),
    "PredictionMode": ("drevalpy.types.enums.prediction_mode", "PredictionMode"),
    "Predictor": ("drevalpy.components.predictors.abstract.base", "Predictor"),
    "ResponseBatch": ("drevalpy.types.data.batch.response_batch", "ResponseBatch"),
    "RunResult": ("drevalpy.types.results", "RunResult"),
    "Section": ("drevalpy.visualization.base", "Section"),
    "SplitMask": ("drevalpy.types.data.split_mask", "SplitMask"),
    "SplitMasks": ("drevalpy.types.data.split_masks", "SplitMasks"),
    "SplitValidationError": ("drevalpy.registry.splitter", "SplitValidationError"),
    "Splitter": ("drevalpy.registry.splitter", "Splitter"),
    "TrainingContext": ("drevalpy.components.contracts.training_context", "TrainingContext"),
    "Validation": ("drevalpy.registry.splitter", "Validation"),
    "Visualization": ("drevalpy.visualization.base", "Visualization"),
    "graph_feature_block": ("drevalpy.types.data.batch.feature_block", "graph_feature_block"),
    "merge_feature_blocks": ("drevalpy.types.data.batch.feature_block", "merge_feature_blocks"),
    "metadata_feature_block": ("drevalpy.types.data.batch.feature_block", "metadata_feature_block"),
    "numeric_feature_block": ("drevalpy.types.data.batch.feature_block", "numeric_feature_block"),
    "ragged_feature_block": ("drevalpy.types.data.batch.feature_block", "ragged_feature_block"),
    "register_cell_line_featurizer": ("drevalpy.registry.cell_line_featurizer", "register"),
    "register_drug_featurizer": ("drevalpy.registry.drug_featurizer", "register"),
    "register_predictor": ("drevalpy.registry.predictor", "register"),
    "register_splitter": ("drevalpy.registry.splitter", "register"),
    "register_visualization": ("drevalpy.registry.visualization", "register"),
}


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


class TestPublicSurface:
    @pytest.mark.parametrize("name", sorted(plugin.__all__))
    def test_every_exported_name_resolves(self, name):
        assert hasattr(plugin, name)

    def test_all_is_sorted_and_unique(self):
        assert list(plugin.__all__) == sorted(set(plugin.__all__))

    def test_nothing_public_is_left_out_of_all(self):
        public = {name for name in vars(plugin) if not name.startswith("_")}
        modules = {name for name in public if isinstance(vars(plugin)[name], type(importlib))}

        assert public - modules == set(plugin.__all__)

    def test_the_facade_defines_nothing_itself(self):
        """A definition here would be a second implementation to keep in sync."""
        own = [name for name in plugin.__all__ if getattr(vars(plugin)[name], "__module__", "") == "drevalpy.plugin"]

        assert own == []


class TestAliasesPointAtTheRealSymbols:
    @pytest.mark.parametrize(("alias", "origin"), sorted(EXPECTED_ORIGINS.items()))
    def test_alias_is_the_same_object(self, alias, origin):
        module_name, attribute = origin
        module = importlib.import_module(module_name)

        assert getattr(plugin, alias) is getattr(module, attribute)

    def test_every_export_has_a_recorded_origin(self):
        """Adding an export without recording where it comes from fails here."""
        assert sorted(EXPECTED_ORIGINS) == sorted(plugin.__all__)

    def test_the_five_register_aliases_are_distinct(self):
        registrars = {
            plugin.register_cell_line_featurizer,
            plugin.register_drug_featurizer,
            plugin.register_predictor,
            plugin.register_splitter,
            plugin.register_visualization,
        }

        assert len(registrars) == 5


class TestPyTypedMarker:
    """PEP 561: without this file, a plugin's type checker treats drevalpy as untyped."""

    def test_the_marker_exists_in_the_source_tree(self):
        assert (REPO_ROOT / "drevalpy" / "py.typed").is_file()

    def test_the_wheel_target_names_the_marker_as_an_artifact(self, pyproject):
        """``py.typed`` is not a ``.py`` file, so hatchling needs it listed."""
        wheel = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]

        assert "drevalpy/py.typed" in wheel["artifacts"]

    @pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to build a wheel")
    def test_the_marker_ships_in_the_built_wheel(self, tmp_path):
        """Built for real: a marker missing from the wheel does nothing for consumers."""
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
            names = archive.namelist()
        assert "drevalpy/py.typed" in names
        assert not [name for name in names if name.startswith(("tests/", "tools/", "docs/"))]


class TestEditableInstallDoesNotLeak:
    """The default editable install puts the bare project root on ``sys.path``.

    That makes drevalpy's own ``tests``, ``tools`` and ``docs`` importable by
    every consumer of the environment, so a plugin repo's ``import tests.x``
    resolved to *drevalpy's* tests. ``dev-mode-exact`` replaces the bare path
    with a finder that maps only the ``drevalpy`` package.
    """

    def test_dev_mode_exact_is_enabled(self, pyproject):
        assert pyproject["tool"]["hatch"]["build"]["dev-mode-exact"] is True

    def test_the_editables_runtime_dependency_is_declared(self, pyproject):
        """``dev-mode-exact`` emits a finder importing ``editables`` at start-up.

        It has to be a real ``[project.dependencies]`` entry, not a dev-group
        one. Dev groups do not propagate to consumers, and uv reads static
        metadata straight from pyproject.toml for path dependencies rather than
        from the built editable wheel that declares ``editables`` itself - so a
        repo installing drevalpy as an editable path dependency got
        ``ModuleNotFoundError: No module named 'editables'`` on every import.
        """
        assert any(entry.startswith("editables") for entry in pyproject["project"]["dependencies"])
        assert not any(entry.startswith("editables") for entry in pyproject["dependency-groups"]["dev"])

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
    def test_importing_only_the_facade_is_enough_to_subclass(self):
        """A plugin importing nothing but the facade must be able to declare a component."""
        script = (
            "from drevalpy.plugin import (\n"
            "    CellLineFeaturizer, FeatureFormat, BlockSpec, numeric_feature_block,\n"
            "    register_cell_line_featurizer,\n"
            ")\n"
            "import numpy as np\n"
            "from typing import ClassVar\n"
            "\n"
            "@register_cell_line_featurizer(\n"
            "    'facadeProbe', description='probe', contract=FeatureFormat.NUMERIC_MATRIX\n"
            ")\n"
            "class Probe(CellLineFeaturizer):\n"
            "    '''Probe.'''\n"
            "    entity_id_only: ClassVar[bool] = True\n"
            "    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (\n"
            "        BlockSpec('probe', FeatureFormat.NUMERIC_MATRIX),\n"
            "    )\n"
            "    def _fit(self, source, **kwargs):\n"
            "        return self\n"
            "    def _transform_blocks(self, source, entity_ids):\n"
            "        return {'probe': numeric_feature_block(np.zeros((len(entity_ids), 1), dtype=np.float32))}\n"
            "    @property\n"
            "    def output_dim(self):\n"
            "        return 1\n"
            "\n"
            "from drevalpy.registry import cell_line_featurizer\n"
            "print('facadeProbe' in cell_line_featurizer.list())\n"
        )

        result = run_trusted_python(script)

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().endswith("True")
