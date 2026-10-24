"""Tests for external component and zoo loading."""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
import textwrap
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest

from drevalpy.errors import InputNotFoundError, InvalidInputError
from drevalpy.models import DRPModel, ModelSpec
from drevalpy.models.config import ModelConfig
from drevalpy.models.zoo import clear_external_zoo, get_zoo_config, list_zoo_names, load_external_zoo_file
from drevalpy.registry._builtins import is_known_builtin_predictor
from drevalpy.registry._extensions import (
    EXTENSIONS_DIR_ENV_VAR,
    _extension_module_name,
    extension_dirs_from_env,
    load_extension_dir,
    load_extension_file,
    load_extension_module,
    load_extensions,
)
from drevalpy.registry.dataset._registry import DatasetRegistry
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.registry.predictor import list as list_predictors
from drevalpy.registry.predictor import predictor_registry
from drevalpy.registry.sample_featurizer import get as get_sample_featurizer
from drevalpy.registry.sample_featurizer import list as list_sample_featurizers
from drevalpy.registry.sample_featurizer import sample_featurizer_registry
from tests._trusted_subprocess import run_trusted_python
from tests.models.synthetic_fixtures import lso_split_masks, synthetic_mudataset
from tests.registry._helpers import restore_component_registries


@pytest.fixture(autouse=True)
def _restore_registries() -> Iterator[None]:
    yield
    restore_component_registries()
    clear_external_zoo()


_SAMPLE_FEATURIZER_SOURCE = """
import numpy as np
from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.sample.base import SampleFeaturizer
from drevalpy.types.data.batch.feature_block import FeaturizerResult, numeric_feature_block
from drevalpy.registry.sample_featurizer import register as register_sample_featurizer


@register_sample_featurizer({name!r}, description="toy", contract=FeatureFormat.NUMERIC_MATRIX)
class ToySampleFeaturizer(SampleFeaturizer):
    entity_id_only = True

    def _fit(self, source, *, entity_ids=None, pair_expanded_ids=None, pair_expanded_es_ids=None):
        self._output_dim = 1
        return self

    def _transform_features(self, source, entity_ids):
        values = np.ones((len(entity_ids), 1), dtype=np.float32)
        return FeaturizerResult(values, {{"toy": numeric_feature_block(values)}})

    @property
    def output_dim(self):
        return self._output_dim
"""

_TOY_PREDICTOR_TEMPLATE = """\
import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
from drevalpy.registry.predictor import register as register_predictor


@register_predictor({name!r}, description='toy',
    sample_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX)
class Toy(FeatureFreePredictor):
    def _fit(self, batch): return None
    def _predict(self, batch): return np.zeros(batch.n_pairs)
"""


def _write_toy_predictor_dir(directory: Path, name: str, filename: str = "toy.py") -> Path:
    directory.mkdir(exist_ok=True)
    (directory / filename).write_text(_TOY_PREDICTOR_TEMPLATE.format(name=name), encoding="utf-8")
    return directory


def test_load_extension_file_registers_components(tmp_path: Path) -> None:
    ext_file = tmp_path / "toy_extension.py"
    ext_file.write_text(
        _SAMPLE_FEATURIZER_SOURCE.format(name="toySample") + _TOY_PREDICTOR_TEMPLATE.format(name="toyPredictor"),
        encoding="utf-8",
    )
    assert "toySample" not in list_sample_featurizers()

    load_extension_file(ext_file)

    assert "toySample" in list_sample_featurizers()
    get_sample_featurizer("toySample")
    get_predictor("toyPredictor")


def test_load_extension_dir_imports_every_file(tmp_path: Path) -> None:
    _write_toy_predictor_dir(tmp_path, "toyB", "b_ext.py")
    _write_toy_predictor_dir(tmp_path, "toyA", "a_ext.py")

    load_extension_dir(tmp_path)

    get_predictor("toyA")
    get_predictor("toyB")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, []),
        ("", []),
        ("/one", ["/one"]),
        (os.pathsep.join(["/one", "/two", "/three"]), ["/one", "/two", "/three"]),
        (os.pathsep + "/one" + os.pathsep * 2, ["/one"]),
    ],
    ids=["unset", "empty", "single", "several-in-order", "empty-entries-dropped"],
)
def test_extension_dirs_from_env(monkeypatch: pytest.MonkeyPatch, value: str | None, expected: list[str]) -> None:
    if value is None:
        monkeypatch.delenv(EXTENSIONS_DIR_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(EXTENSIONS_DIR_ENV_VAR, value)

    assert extension_dirs_from_env() == expected


class TestLoadExtensionsEnvVar:
    def test_every_env_directory_is_loaded(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        first = _write_toy_predictor_dir(tmp_path / "first", "envToyA")
        second = _write_toy_predictor_dir(tmp_path / "second", "envToyB")
        monkeypatch.setenv(EXTENSIONS_DIR_ENV_VAR, os.pathsep.join([str(first), str(second)]))

        load_extensions()

        get_predictor("envToyA")
        get_predictor("envToyB")

    @pytest.mark.parametrize("with_flag_dir", [True, False], ids=["flag-dir", "empty-list"])
    def test_explicit_directories_do_not_read_the_env_var(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, with_flag_dir: bool
    ) -> None:
        env_dir = _write_toy_predictor_dir(tmp_path / "env", "envToy")
        monkeypatch.setenv(EXTENSIONS_DIR_ENV_VAR, str(env_dir))
        directories = [_write_toy_predictor_dir(tmp_path / "flag", "flagToy")] if with_flag_dir else []

        load_extensions(directories=directories)

        assert "envToy" not in list_predictors()
        assert ("flagToy" in list_predictors()) is with_flag_dir

    def test_missing_env_directory_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv(EXTENSIONS_DIR_ENV_VAR, str(tmp_path / "missing"))

        with pytest.raises(InputNotFoundError, match="Extension directory not found"):
            load_extensions()


class TestRepeatedLoads:
    def test_loading_the_same_directory_file_and_env_twice_is_harmless(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        ext_dir = _write_toy_predictor_dir(tmp_path / "ext", "twiceToy")
        monkeypatch.setenv(EXTENSIONS_DIR_ENV_VAR, str(ext_dir))

        load_extension_dir(ext_dir)
        load_extension_dir(ext_dir)
        load_extension_file(ext_dir / "toy.py")
        load_extensions()
        load_extensions(directories=[ext_dir])

        get_predictor("twiceToy")

    def test_file_is_executed_again_after_its_registrations_were_evicted(self, tmp_path: Path) -> None:
        ext_dir = _write_toy_predictor_dir(tmp_path / "ext", "evictedToy")
        load_extension_dir(ext_dir)
        restore_component_registries()
        assert "evictedToy" not in list_predictors()

        load_extension_dir(ext_dir)

        get_predictor("evictedToy")


def test_loading_the_same_dataset_declaration_yaml_twice_is_harmless(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DREVALPY_CONFIG_DIR", str(tmp_path / "config"))
    registry = DatasetRegistry()
    monkeypatch.setattr("drevalpy.registry.dataset._registry.dataset_registry", registry)
    ext_dir = tmp_path / "ext"
    ext_dir.mkdir()
    (ext_dir / "datasets.yaml").write_text(
        "sources:\n  mirror:\n    url: file:///data\ndatasets:\n  Toy:\n    source: mirror\n    file: toy.h5mu\n",
        encoding="utf-8",
    )

    load_extension_dir(ext_dir)
    load_extension_dir(ext_dir)

    assert registry.custom_datasets["Toy"].file == "toy.h5mu"


_MEAN_PREDICTOR_SOURCE = """
import numpy as np
from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
from drevalpy.registry.predictor import register as register_predictor


@register_predictor(
    "externalPredictor",
    description="ext",
    sample_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class ExternalPredictor(FeatureFreePredictor):
    def _fit(self, batch):
        self._mean = float(np.mean(batch.response))

    def _predict(self, batch):
        return np.full(batch.n_pairs, self._mean, dtype=np.float64)

    def get_state(self):
        return {"mean": self._mean} if hasattr(self, "_mean") else {}

    def set_state(self, state):
        if "mean" in state:
            self._mean = float(state["mean"])

    def is_fitted(self):
        return hasattr(self, "_mean")
"""


def test_external_zoo_references_extension_components(tmp_path: Path) -> None:
    ext_dir = tmp_path / "ext"
    ext_dir.mkdir()
    (ext_dir / "components.py").write_text(
        _SAMPLE_FEATURIZER_SOURCE.format(name="externalSample") + _MEAN_PREDICTOR_SOURCE, encoding="utf-8"
    )
    zoo_file = tmp_path / "external_zoo.yaml"
    zoo_file.write_text("externalToy:\n  predictor: externalPredictor\n", encoding="utf-8")

    load_extensions(directories=[ext_dir], zoo_files=[zoo_file])

    assert "externalToy" in list_zoo_names(include_external=True)
    model = DRPModel(ModelSpec("externalToy", get_zoo_config("externalToy")))
    dataset, masks = synthetic_mudataset(), lso_split_masks()
    model.train(dataset, masks.train)
    assert np.isfinite(model.predict(dataset, masks.test)).all()


def test_load_external_zoo_single_entry_file(tmp_path: Path) -> None:
    zoo_file = tmp_path / "single.yaml"
    zoo_file.write_text("name: customNaive\npredictor: naiveMean\n", encoding="utf-8")

    assert load_external_zoo_file(zoo_file) == ["customNaive"]

    config = get_zoo_config("customNaive")
    assert isinstance(config, ModelConfig)
    assert config.predictor.name == "naiveMean"


def test_load_external_zoo_invalid_entry_reports_path(tmp_path: Path) -> None:
    zoo_file = tmp_path / "bad_zoo.yaml"
    zoo_file.write_text("brokenEntry:\n  predictor: naiveMean\n  unknown_key: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="brokenEntry"):
        load_external_zoo_file(zoo_file)


def test_extension_module_name_uses_stable_path_digest(tmp_path: Path) -> None:
    ext_file = tmp_path / "stable.py"
    ext_file.write_text("# noop\n", encoding="utf-8")
    resolved = ext_file.resolve()
    digest = hashlib.sha256(str(resolved).encode()).hexdigest()[:16]

    assert _extension_module_name(resolved) == f"drevalpy_user_extension_stable_{digest}"


def test_failed_extension_file_does_not_leave_sys_modules_or_registry_mutation(tmp_path: Path) -> None:
    ext_file = tmp_path / "broken_extension.py"
    ext_file.write_text(
        _TOY_PREDICTOR_TEMPLATE.format(name="brokenPartial") + '\nraise RuntimeError("fail after registration")\n',
        encoding="utf-8",
    )
    module_name = _extension_module_name(ext_file.resolve())
    before_predictors = set(list_predictors())

    with pytest.raises(ImportError, match="broken_extension.py"):
        load_extension_file(ext_file)

    assert module_name not in sys.modules
    assert set(list_predictors()) == before_predictors


@pytest.fixture
def extension_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[str, str], str]]:
    """Write an importable module into ``tmp_path`` and forget it from ``sys.modules`` afterwards."""
    monkeypatch.syspath_prepend(tmp_path)
    created: list[str] = []

    def write(name: str, source: str) -> str:
        (tmp_path / f"{name}.py").write_text(source, encoding="utf-8")
        importlib.invalidate_caches()
        created.append(name)
        return name

    yield write
    for name in created:
        sys.modules.pop(name, None)


class TestLoadExtensionModule:
    def test_load_extensions_imports_modules_and_files_so_their_decorators_run(
        self, tmp_path: Path, extension_module: Callable[[str, str], str]
    ) -> None:
        module = extension_module("ext_toy_module", _TOY_PREDICTOR_TEMPLATE.format(name="moduleToy"))
        ext_file = _write_toy_predictor_dir(tmp_path / "files", "fileToy") / "toy.py"

        load_extensions(modules=[module], files=[ext_file], directories=[])

        get_predictor("moduleToy")
        get_predictor("fileToy")

    def test_an_empty_module_name_is_rejected(self) -> None:
        with pytest.raises(InvalidInputError, match="non-empty"):
            load_extension_module("")

    @pytest.mark.parametrize(
        ("raised", "message"),
        [("ImportError('missing dependency')", "missing dependency"), ("RuntimeError('boom')", "Failed to import")],
        ids=["import-error-propagates", "other-error-is-wrapped"],
    )
    def test_a_module_failing_after_registering_leaves_the_registries_untouched(
        self, extension_module: Callable[[str, str], str], raised: str, message: str
    ) -> None:
        module = extension_module(
            "ext_failing_module", _TOY_PREDICTOR_TEMPLATE.format(name="partialModule") + f"\nraise {raised}\n"
        )
        before = set(list_predictors())

        with pytest.raises(ImportError, match=message):
            load_extension_module(module)

        assert set(list_predictors()) == before


class TestLoadExtensionFileErrors:
    def test_a_missing_file_raises_input_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(InputNotFoundError, match="Extension file not found"):
            load_extension_file(tmp_path / "absent.py")

    def test_a_file_python_cannot_import_is_rejected(self, tmp_path: Path) -> None:
        not_python = tmp_path / "notes.txt"
        not_python.write_text("print('hi')\n", encoding="utf-8")

        with pytest.raises(ImportError, match="Could not load extension file"):
            load_extension_file(not_python)

    def test_an_import_error_in_the_file_propagates_and_rolls_back_its_registrations(self, tmp_path: Path) -> None:
        ext_file = tmp_path / "needs_missing.py"
        ext_file.write_text(
            _TOY_PREDICTOR_TEMPLATE.format(name="beforeImportError") + "\nimport drevalpy_no_such_dependency\n",
            encoding="utf-8",
        )
        module_name = _extension_module_name(ext_file.resolve())

        with pytest.raises(ImportError, match="drevalpy_no_such_dependency"):
            load_extension_file(ext_file)

        assert module_name not in sys.modules
        assert "beforeImportError" not in list_predictors()


class TestLoadExtensionDirErrors:
    def test_a_missing_directory_raises_input_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(InputNotFoundError, match="Extension directory not found"):
            load_extension_dir(tmp_path / "absent")

    def test_a_zoo_yaml_in_the_directory_is_dispatched_to_the_zoo_loader_and_other_yaml_is_ignored(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "a_zoo.yaml").write_text("name: dirZoo\npredictor: naiveMean\n", encoding="utf-8")
        (tmp_path / "b_list.yaml").write_text("- not\n- a mapping\n", encoding="utf-8")

        load_extension_dir(tmp_path)

        assert "dirZoo" in list_zoo_names(include_external=True)

    def test_a_bad_file_rolls_back_what_the_directory_registered_before_it(self, tmp_path: Path) -> None:
        _write_toy_predictor_dir(tmp_path, "rolledBackToy", "a_good.py")
        (tmp_path / "z_bad.yaml").write_text(
            "brokenEntry:\n  predictor: naiveMean\n  unknown_key: 1\n", encoding="utf-8"
        )

        with pytest.raises(ValueError, match="brokenEntry"):
            load_extension_dir(tmp_path)

        assert "rolledBackToy" not in list_predictors()

    def test_a_name_already_registered_by_another_file_is_a_duplicate_error_and_rolls_back(
        self, tmp_path: Path
    ) -> None:
        _write_toy_predictor_dir(tmp_path, "dupToy", "a_first.py")
        _write_toy_predictor_dir(tmp_path, "dupToy", "b_second.py")

        with pytest.raises(ImportError, match="b_second.py"):
            load_extension_dir(tmp_path)

        assert "dupToy" not in list_predictors()


def test_is_known_builtin_predictor_public_query() -> None:
    assert is_known_builtin_predictor("elasticNet")
    assert not is_known_builtin_predictor("notARealPredictor")


_BLOCK_OPTIONAL_FAMILIES = textwrap.dedent("""
    import importlib.abc
    import importlib.machinery
    import sys

    blocked = {
        "xgboost": "blocked xgboost",
        "lightgbm": "blocked lightgbm",
        "drevalpy.components.predictors.literature.dipk.predictor": "blocked dipk",
    }

    class BlockLoader(importlib.abc.Loader):
        def __init__(self, message: str) -> None:
            self.message = message

        def create_module(self, spec):
            raise ImportError(self.message)

        def exec_module(self, module):
            raise ImportError(self.message)

    class BlockFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname in blocked:
                return importlib.machinery.ModuleSpec(fullname, BlockLoader(blocked[fullname]))
            return None

    sys.meta_path.insert(0, BlockFinder())
    """)


class TestSubprocessImportIsolation:
    """Each test spawns a fresh interpreter: a shared one would let the first import graph answer for the second."""

    pytestmark = pytest.mark.slow

    def test_subprocess_extension_load_does_not_import_optional_families(self, tmp_path: Path) -> None:
        ext_dir = _write_toy_predictor_dir(tmp_path / "ext", "isolatedPredictor")
        script = _BLOCK_OPTIONAL_FAMILIES + textwrap.dedent(f"""
            from drevalpy.registry._extensions import load_extension_file
            from drevalpy.registry.predictor import get as get_predictor

            load_extension_file({str(ext_dir / "toy.py")!r})
            get_predictor("isolatedPredictor")
            """)
        completed = run_trusted_python(script)
        assert completed.returncode == 0, completed.stdout + completed.stderr

    def test_subprocess_native_lookup_does_not_import_optional_families(self) -> None:
        script = _BLOCK_OPTIONAL_FAMILIES + textwrap.dedent("""
            from drevalpy.registry.sample_featurizer import get as get_sample_featurizer
            from drevalpy.registry.predictor import get as get_predictor

            get_sample_featurizer("identity")
            get_predictor("elasticNet")
            """)
        completed = run_trusted_python(script)
        assert completed.returncode == 0, completed.stdout + completed.stderr


def test_unknown_builtin_predictor_raises_value_error() -> None:
    predictor_registry.clear()
    with pytest.raises(ValueError, match="Unknown Predictor"):
        get_predictor("notRegisteredAnywhere")


def test_unknown_builtin_featurizer_raises_value_error() -> None:
    sample_featurizer_registry.clear()
    with pytest.raises(ValueError, match="Unknown Sample featurizer"):
        get_sample_featurizer("notRegisteredAnywhere")
