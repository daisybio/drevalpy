"""Tests for external component and zoo loading."""

from __future__ import annotations

import hashlib
import os
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from drevalpy.errors import InputNotFoundError
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
    load_extensions,
)
from drevalpy.registry.dataset._registry import DatasetRegistry
from drevalpy.registry.predictor import (
    get as get_predictor,
)
from drevalpy.registry.predictor import (
    list as list_predictors,
)
from drevalpy.registry.predictor import predictor_registry
from drevalpy.registry.sample_featurizer import (
    get as get_sample_featurizer,
)
from drevalpy.registry.sample_featurizer import (
    list as list_sample_featurizers,
)
from drevalpy.registry.sample_featurizer import sample_featurizer_registry
from tests._trusted_subprocess import run_trusted_python
from tests.registry._helpers import restore_component_registries


@pytest.fixture(autouse=True)
def _restore_registries() -> Iterator[None]:
    """Evict whatever a test in this file registered into the process-global state.

    Nearly every test here loads an extension, and loading an extension registers
    into the component registries and the external zoo for the whole process.
    ``tests/conftest.py`` repopulates the built-ins before each test but
    ``register_builtin_components`` only ever *adds* - it does not evict - so
    without this teardown the extensions outlive their tests. The
    ``BUILTIN_*_NAMES`` sets in :mod:`drevalpy.registry._builtins` are lazy
    singletons that cache on first access, so whichever test resolves them next
    counts a leaked extension as a built-in, and
    ``tests/docs/test_docs_structure.py`` fails whenever it happens to run after
    this file rather than before it. Mirrors the fixture in
    ``tests/models/config/test_spec.py``.
    """
    yield
    restore_component_registries()
    clear_external_zoo()


def test_load_extension_file_registers_components(tmp_path: Path) -> None:
    ext_file = tmp_path / "toy_extension.py"
    ext_file.write_text(
        """
from __future__ import annotations

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.sample.base import SampleFeaturizer
from drevalpy.types.data.batch.feature_block import FeaturizerResult, numeric_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
from drevalpy.registry.predictor import register
from drevalpy.registry.sample_featurizer import register as register_sample_featurizer


@register_sample_featurizer(
    "toySample",
    description="Toy featurizer",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class ToySampleFeaturizer(SampleFeaturizer):
    entity_id_only = True

    def _fit(self, source, *, entity_ids=None, pair_expanded_ids=None, pair_expanded_es_ids=None):
        self._output_dim = 1
        return self

    def _transform_features(self, source, entity_ids):
        values = np.ones((len(entity_ids), 1), dtype=np.float32)
        return FeaturizerResult(values, {"toy": numeric_feature_block(values)})

    def _empty_transform_features(self, source):
        values = np.empty((0, 1), dtype=np.float32)
        return FeaturizerResult(values, {"toy": numeric_feature_block(values)})

    @property
    def output_dim(self):
        return self._output_dim


@register(
    "toyPredictor",
    description="Toy predictor",
    sample_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class ToyPredictor(FeatureFreePredictor):
    def _fit(self, batch: ModelInputBatch) -> None:
        return None

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        return np.zeros(batch.n_pairs, dtype=np.float64)
""",
        encoding="utf-8",
    )
    before = set(list_sample_featurizers())
    load_extension_file(ext_file)
    assert "toySample" in list_sample_featurizers()
    assert "toySample" not in before
    get_sample_featurizer("toySample")
    get_predictor("toyPredictor")


def test_load_extension_dir_imports_sorted_files(tmp_path: Path) -> None:
    (tmp_path / "b_ext.py").write_text(
        "from drevalpy.registry.predictor import register as register_predictor\n"
        "from drevalpy.components.contracts.contracts import FeatureFormat\n"
        "from drevalpy.types.data.batch.model_input_batch import ModelInputBatch\n"
        "from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor\n"
        "import numpy as np\n"
        "@register_predictor('toyB', description='b',\n"
        "    sample_contract=FeatureFormat.NUMERIC_MATRIX,\n"
        "    drug_contract=FeatureFormat.NUMERIC_MATRIX)\n"
        "class ToyB(FeatureFreePredictor):\n"
        "    def _fit(self, batch: ModelInputBatch): return None\n"
        "    def _predict(self, batch: ModelInputBatch): return np.zeros(batch.n_pairs)\n",
        encoding="utf-8",
    )
    (tmp_path / "a_ext.py").write_text(
        "from drevalpy.registry.predictor import register as register_predictor\n"
        "from drevalpy.components.contracts.contracts import FeatureFormat\n"
        "from drevalpy.types.data.batch.model_input_batch import ModelInputBatch\n"
        "from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor\n"
        "import numpy as np\n"
        "@register_predictor('toyA', description='a',\n"
        "    sample_contract=FeatureFormat.NUMERIC_MATRIX,\n"
        "    drug_contract=FeatureFormat.NUMERIC_MATRIX)\n"
        "class ToyA(FeatureFreePredictor):\n"
        "    def _fit(self, batch: ModelInputBatch): return None\n"
        "    def _predict(self, batch: ModelInputBatch): return np.zeros(batch.n_pairs)\n",
        encoding="utf-8",
    )
    load_extension_dir(tmp_path)
    get_predictor("toyA")
    get_predictor("toyB")


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


def _write_toy_predictor_dir(directory: Path, name: str) -> Path:
    directory.mkdir()
    (directory / "toy.py").write_text(_TOY_PREDICTOR_TEMPLATE.format(name=name), encoding="utf-8")
    return directory


class TestExtensionDirsFromEnv:
    def test_unset_gives_no_directories(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(EXTENSIONS_DIR_ENV_VAR, raising=False)

        assert extension_dirs_from_env() == []

    def test_empty_gives_no_directories(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(EXTENSIONS_DIR_ENV_VAR, "")

        assert extension_dirs_from_env() == []

    def test_single_directory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(EXTENSIONS_DIR_ENV_VAR, "/one")

        assert extension_dirs_from_env() == ["/one"]

    def test_several_directories_split_on_pathsep_in_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(EXTENSIONS_DIR_ENV_VAR, os.pathsep.join(["/one", "/two", "/three"]))

        assert extension_dirs_from_env() == ["/one", "/two", "/three"]

    def test_empty_entries_are_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(EXTENSIONS_DIR_ENV_VAR, os.pathsep + "/one" + os.pathsep * 2)

        assert extension_dirs_from_env() == ["/one"]


class TestLoadExtensionsEnvVar:
    def test_none_directories_load_the_env_directory(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        ext_dir = _write_toy_predictor_dir(tmp_path / "ext", "envToy")
        monkeypatch.setenv(EXTENSIONS_DIR_ENV_VAR, str(ext_dir))

        load_extensions()

        get_predictor("envToy")

    def test_several_env_directories_are_all_loaded(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        first = _write_toy_predictor_dir(tmp_path / "first", "envToyA")
        second = _write_toy_predictor_dir(tmp_path / "second", "envToyB")
        monkeypatch.setenv(EXTENSIONS_DIR_ENV_VAR, os.pathsep.join([str(first), str(second)]))

        load_extensions()

        get_predictor("envToyA")
        get_predictor("envToyB")

    def test_explicit_directories_do_not_read_the_env_var(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        env_dir = _write_toy_predictor_dir(tmp_path / "env", "envToy")
        flag_dir = _write_toy_predictor_dir(tmp_path / "flag", "flagToy")
        monkeypatch.setenv(EXTENSIONS_DIR_ENV_VAR, str(env_dir))

        load_extensions(directories=[flag_dir])

        get_predictor("flagToy")
        assert "envToy" not in list_predictors()

    def test_empty_explicit_directories_do_not_read_the_env_var(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        env_dir = _write_toy_predictor_dir(tmp_path / "env", "envToy")
        monkeypatch.setenv(EXTENSIONS_DIR_ENV_VAR, str(env_dir))

        load_extensions(directories=[])

        assert "envToy" not in list_predictors()

    def test_missing_env_directory_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv(EXTENSIONS_DIR_ENV_VAR, str(tmp_path / "missing"))

        with pytest.raises(InputNotFoundError, match="Extension directory not found"):
            load_extensions()


class TestRepeatedLoads:
    def test_loading_the_same_directory_twice_is_harmless(self, tmp_path: Path) -> None:
        ext_dir = _write_toy_predictor_dir(tmp_path / "ext", "twiceToy")

        load_extension_dir(ext_dir)
        load_extension_dir(ext_dir)

        get_predictor("twiceToy")

    def test_loading_the_same_file_twice_is_harmless(self, tmp_path: Path) -> None:
        ext_dir = _write_toy_predictor_dir(tmp_path / "ext", "twiceToy")

        load_extension_file(ext_dir / "toy.py")
        load_extension_file(ext_dir / "toy.py")

        get_predictor("twiceToy")

    def test_env_and_flag_naming_the_same_directory_is_harmless(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        ext_dir = _write_toy_predictor_dir(tmp_path / "ext", "twiceToy")
        monkeypatch.setenv(EXTENSIONS_DIR_ENV_VAR, str(ext_dir))

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


def test_external_zoo_references_extension_components(tmp_path: Path) -> None:
    ext_dir = tmp_path / "ext"
    ext_dir.mkdir()
    (ext_dir / "components.py").write_text(
        """
import numpy as np
from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.sample.base import SampleFeaturizer
from drevalpy.types.data.batch.feature_block import FeaturizerResult, numeric_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
from drevalpy.registry.predictor import register as register_predictor
from drevalpy.registry.sample_featurizer import register as register_sample_featurizer

@register_sample_featurizer(
    "externalSample",
    description="ext",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class ExternalSampleFeaturizer(SampleFeaturizer):
    entity_id_only = True

    def _fit(self, source, *, entity_ids=None, pair_expanded_ids=None, pair_expanded_es_ids=None):
        self._output_dim = 1
        return self
    def _transform_features(self, source, entity_ids):
        values = np.ones((len(entity_ids), 1), dtype=np.float32)
        return FeaturizerResult(values, {"ext": numeric_feature_block(values)})
    def _empty_transform_features(self, source):
        values = np.empty((0, 1), dtype=np.float32)
        return FeaturizerResult(values, {"ext": numeric_feature_block(values)})
    @property
    def output_dim(self):
        return self._output_dim

@register_predictor(
    "externalPredictor",
    description="ext",
    sample_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class ExternalPredictor(FeatureFreePredictor):
    def _fit(self, batch: ModelInputBatch) -> None:
        if batch.response is None:
            msg = "response required"
            raise ValueError(msg)
        self._mean = float(np.mean(batch.response))

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        return np.full(batch.n_pairs, self._mean, dtype=np.float64)

    def get_state(self) -> dict[str, object]:
        if not hasattr(self, "_mean"):
            return {}
        return {"mean": self._mean}

    def set_state(self, state: dict[str, object]) -> None:
        if "mean" in state:
            self._mean = float(state["mean"])

    def is_fitted(self) -> bool:
        return hasattr(self, "_mean")
""",
        encoding="utf-8",
    )
    zoo_file = tmp_path / "external_zoo.yaml"
    zoo_file.write_text(
        """
externalToy:
  predictor: externalPredictor
""",
        encoding="utf-8",
    )
    load_extensions(directories=[ext_dir], zoo_files=[zoo_file])
    assert "externalToy" in list_zoo_names(include_external=True)
    config = get_zoo_config("externalToy")
    model = DRPModel(ModelSpec("externalToy", config))

    import anndata as ad
    import mudata as md
    import pandas as pd

    from drevalpy.types import SplitMask, SplitMasks
    from drevalpy.types.data.dataset import Dataset

    cl_ids = np.array(["cl1", "cl2"])
    drug_ids = np.array(["d1", "d2"])
    response_matrix = np.array([[1.0, 1.0], [3.0, 3.0]], dtype=np.float32)
    response_ad = ad.AnnData(
        X=response_matrix,
        obs=pd.DataFrame({"cell_line_name": cl_ids, "tissue": ["L", "B"]}, index=cl_ids),
        var=pd.DataFrame(index=drug_ids),
    )
    mudataset = Dataset(md.MuData({"response": response_ad}), name="test")
    split = SplitMasks(
        train=SplitMask(np.array([[True, True], [False, False]])),
        test=SplitMask(np.array([[False, False], [True, True]])),
        val=SplitMask(np.zeros((2, 2), dtype=bool)),
    )
    model.train(mudataset, split.train)
    preds = model.predict(mudataset, split.test)
    assert np.isfinite(preds).all()


def test_load_external_zoo_single_entry_file(tmp_path: Path) -> None:
    zoo_file = tmp_path / "single.yaml"
    zoo_file.write_text(
        """
name: customNaive
predictor: naiveMean
""",
        encoding="utf-8",
    )
    names = load_external_zoo_file(zoo_file)
    assert names == ["customNaive"]
    config = get_zoo_config("customNaive")
    assert isinstance(config, ModelConfig)
    assert config.predictor.name == "naiveMean"


def test_load_external_zoo_invalid_entry_reports_path(tmp_path: Path) -> None:
    zoo_file = tmp_path / "bad_zoo.yaml"
    zoo_file.write_text(
        """
brokenEntry:
  predictor: naiveMean
  unknown_key: true
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="brokenEntry"):
        load_external_zoo_file(zoo_file)


def test_extension_module_name_uses_stable_path_digest(tmp_path: Path) -> None:
    ext_file = tmp_path / "stable.py"
    ext_file.write_text("# noop\n", encoding="utf-8")
    resolved = ext_file.resolve()
    expected_digest = hashlib.sha256(str(resolved).encode()).hexdigest()[:16]
    assert _extension_module_name(resolved) == f"drevalpy_user_extension_stable_{expected_digest}"


def test_failed_extension_file_does_not_leave_sys_modules_or_registry_mutation(tmp_path: Path) -> None:
    ext_file = tmp_path / "broken_extension.py"
    ext_file.write_text(
        """
from drevalpy.registry.predictor import register as register_predictor
from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
import numpy as np

@register_predictor(
    "brokenPartial",
    description="partial",
    sample_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class BrokenPartial(FeatureFreePredictor):
    def _fit(self, batch: ModelInputBatch) -> None:
        return None

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        return np.zeros(batch.n_pairs)

raise RuntimeError("fail after partial registration")
""",
        encoding="utf-8",
    )
    module_name = _extension_module_name(ext_file.resolve())
    before_predictors = set(list_predictors())
    with pytest.raises(ImportError, match="broken_extension.py"):
        load_extension_file(ext_file)
    assert module_name not in sys.modules
    assert set(list_predictors()) == before_predictors


def test_is_known_builtin_predictor_public_query() -> None:
    assert is_known_builtin_predictor("elasticNet")
    assert not is_known_builtin_predictor("notARealPredictor")


#: Installs a meta-path finder that turns any import of an optional predictor
#: family into an ``ImportError``, so a code path that reaches for one fails
#: loudly instead of silently pulling in xgboost, lightgbm or dipk.
#:
#: The two tests below share this preamble but deliberately **not** a process.
#: Each asserts that a particular pristine entry point - loading a user extension
#: file, and looking up a native component - stays clear of the optional
#: families, and a shared interpreter would let the first one's import graph
#: answer for the second. That is why both are in the extended tier rather than
#: merged into one child: two interpreter spawns are the test.
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
    """Extended tier: each test spawns a fresh interpreter (~2s each).

    They are not merged into one child process on purpose - see the comment on
    ``_BLOCK_OPTIONAL_FAMILIES`` above. The class exists so the marker applies at
    class level while the other tests in this file stay in the fast tier.
    """

    pytestmark = pytest.mark.slow

    def test_subprocess_extension_load_does_not_import_optional_families(self, tmp_path: Path) -> None:
        ext_file = tmp_path / "isolated_extension.py"
        ext_file.write_text(
            """
from drevalpy.registry.predictor import register as register_predictor
from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
import numpy as np

@register_predictor(
    "isolatedPredictor",
    description="isolated",
    sample_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class IsolatedPredictor(FeatureFreePredictor):
    def _fit(self, batch: ModelInputBatch) -> None:
        return None

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        return np.zeros(batch.n_pairs, dtype=np.float64)
""",
            encoding="utf-8",
        )
        script = _BLOCK_OPTIONAL_FAMILIES + textwrap.dedent(f"""
            from drevalpy.registry._extensions import load_extension_file
            from drevalpy.registry.predictor import get as get_predictor

            load_extension_file({str(ext_file)!r})
            cls = get_predictor("isolatedPredictor")
            assert cls.__name__ == "IsolatedPredictor"
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
