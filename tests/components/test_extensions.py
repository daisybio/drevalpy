"""Tests for external component and zoo loading."""

from __future__ import annotations

import hashlib
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from drevalpy.components.extensions import (
    _extension_module_name,
    load_extension_dir,
    load_extension_file,
    load_extensions,
)
from drevalpy.components.register_builtins import is_known_builtin_predictor
from drevalpy.components.registry import (
    get_cell_line_featurizer,
    get_predictor,
    list_cell_line_featurizers,
    list_predictors,
)
from drevalpy.components.registry.featurizer_registry import cell_line_featurizer_registry
from drevalpy.components.registry.predictor_registry import predictor_registry
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models import construct_model
from drevalpy.models.config import ModelConfig
from drevalpy.models.zoo import get_zoo_config, list_zoo_names, load_external_zoo_file
from tests._trusted_subprocess import run_trusted_python


def test_load_extension_file_registers_components(tmp_path: Path) -> None:
    ext_file = tmp_path / "toy_extension.py"
    ext_file.write_text(
        """
from __future__ import annotations

import numpy as np

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.feature_free import FeatureFreePredictor
from drevalpy.components.registry import register_cell_line_featurizer, register_predictor


@register_cell_line_featurizer(
    "toyCellLine",
    description="Toy featurizer",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class ToyCellLineFeaturizer(CellLineFeaturizer):
    def fit(self, features, *, entity_ids=None):
        self._output_dim = 1
        return self

    def transform(self, features, entity_ids):
        return np.ones((len(entity_ids), 1), dtype=np.float32)

    @property
    def output_dim(self):
        return self._output_dim


@register_predictor(
    "toyPredictor",
    description="Toy predictor",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class ToyPredictor(FeatureFreePredictor):
    def fit(self, batch: ModelInputBatch) -> None:
        return None

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        return np.zeros(batch.n_pairs, dtype=np.float64)
""",
        encoding="utf-8",
    )
    before = set(list_cell_line_featurizers())
    load_extension_file(ext_file)
    assert "toyCellLine" in list_cell_line_featurizers()
    assert "toyCellLine" not in before
    get_cell_line_featurizer("toyCellLine")
    get_predictor("toyPredictor")


def test_load_extension_dir_imports_sorted_files(tmp_path: Path) -> None:
    (tmp_path / "b_ext.py").write_text(
        "from drevalpy.components.registry import register_predictor\n"
        "from drevalpy.components.contracts import FeatureFormat\n"
        "from drevalpy.components.model_input_batch import ModelInputBatch\n"
        "from drevalpy.components.predictors.feature_free import FeatureFreePredictor\n"
        "import numpy as np\n"
        "@register_predictor('toyB', description='b',\n"
        "    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,\n"
        "    drug_contract=FeatureFormat.NUMERIC_MATRIX)\n"
        "class ToyB(FeatureFreePredictor):\n"
        "    def fit(self, batch: ModelInputBatch): return None\n"
        "    def predict(self, batch: ModelInputBatch): return np.zeros(batch.n_pairs)\n",
        encoding="utf-8",
    )
    (tmp_path / "a_ext.py").write_text(
        "from drevalpy.components.registry import register_predictor\n"
        "from drevalpy.components.contracts import FeatureFormat\n"
        "from drevalpy.components.model_input_batch import ModelInputBatch\n"
        "from drevalpy.components.predictors.feature_free import FeatureFreePredictor\n"
        "import numpy as np\n"
        "@register_predictor('toyA', description='a',\n"
        "    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,\n"
        "    drug_contract=FeatureFormat.NUMERIC_MATRIX)\n"
        "class ToyA(FeatureFreePredictor):\n"
        "    def fit(self, batch: ModelInputBatch): return None\n"
        "    def predict(self, batch: ModelInputBatch): return np.zeros(batch.n_pairs)\n",
        encoding="utf-8",
    )
    load_extension_dir(tmp_path)
    get_predictor("toyA")
    get_predictor("toyB")


def test_external_zoo_references_extension_components(tmp_path: Path) -> None:
    ext_dir = tmp_path / "ext"
    ext_dir.mkdir()
    (ext_dir / "components.py").write_text(
        """
import numpy as np
from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.feature_free import FeatureFreePredictor
from drevalpy.components.registry import register_cell_line_featurizer, register_predictor

@register_cell_line_featurizer(
    "externalCellLine",
    description="ext",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class ExternalCellLineFeaturizer(CellLineFeaturizer):
    def fit(self, features, *, entity_ids=None):
        self._output_dim = 1
        return self
    def transform(self, features, entity_ids):
        return np.ones((len(entity_ids), 1), dtype=np.float32)
    @property
    def output_dim(self):
        return self._output_dim

@register_predictor(
    "externalPredictor",
    description="ext",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class ExternalPredictor(FeatureFreePredictor):
    def fit(self, batch: ModelInputBatch) -> None:
        if batch.response is None:
            msg = "response required"
            raise ValueError(msg)
        self._mean = float(np.mean(batch.response))

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
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
    model = construct_model("externalToy", config)()
    response = DrugResponseDataset(
        response=np.array([1.0, 3.0]),
        cell_line_ids=np.array(["cl1", "cl2"]),
        drug_ids=np.array(["d1", "d2"]),
    )
    model.train(response, FeatureDataset(features={"cl1": {}, "cl2": {}}), None)
    preds = model.predict(
        response.cell_line_ids,
        response.drug_ids,
        FeatureDataset(features={"cl1": {}, "cl2": {}}),
        None,
    )
    assert np.allclose(preds, 2.0)


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
from drevalpy.components.registry import register_predictor
from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.feature_free import FeatureFreePredictor
import numpy as np

@register_predictor(
    "brokenPartial",
    description="partial",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class BrokenPartial(FeatureFreePredictor):
    def fit(self, batch: ModelInputBatch) -> None:
        return None

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
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


def test_subprocess_extension_load_does_not_import_optional_families(tmp_path: Path) -> None:
    ext_file = tmp_path / "isolated_extension.py"
    ext_file.write_text(
        """
from drevalpy.components.registry import register_predictor
from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.feature_free import FeatureFreePredictor
import numpy as np

@register_predictor(
    "isolatedPredictor",
    description="isolated",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class IsolatedPredictor(FeatureFreePredictor):
    def fit(self, batch: ModelInputBatch) -> None:
        return None

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        return np.zeros(batch.n_pairs, dtype=np.float64)
""",
        encoding="utf-8",
    )
    script = textwrap.dedent(f"""
        import importlib.abc
        import importlib.machinery
        import sys

        blocked = {{
            "xgboost": "blocked xgboost",
            "lightgbm": "blocked lightgbm",
            "drevalpy.components.predictors.literature.dipk.predictor": "blocked dipk",
        }}

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

        from drevalpy.components.extensions import load_extension_file
        from drevalpy.components.registry import get_predictor

        load_extension_file({str(ext_file)!r})
        cls = get_predictor("isolatedPredictor")
        assert cls.__name__ == "IsolatedPredictor"
        """)
    completed = run_trusted_python(script)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_subprocess_native_lookup_does_not_import_optional_families() -> None:
    script = textwrap.dedent("""
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

        from drevalpy.components.registry import get_cell_line_featurizer, get_predictor

        get_cell_line_featurizer("identity")
        get_predictor("elasticNet")
        """)
    completed = run_trusted_python(script)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_unknown_builtin_predictor_raises_value_error() -> None:
    predictor_registry.clear()
    try:
        with pytest.raises(ValueError, match="Unknown Predictor"):
            get_predictor("notRegisteredAnywhere")
    finally:
        from drevalpy.components.register_builtins import ensure_components_registered

        predictor_registry.clear()
        ensure_components_registered()


def test_unknown_builtin_featurizer_raises_value_error() -> None:
    cell_line_featurizer_registry.clear()
    try:
        with pytest.raises(ValueError, match="Unknown Cell line featurizer"):
            get_cell_line_featurizer("notRegisteredAnywhere")
    finally:
        from drevalpy.components.register_builtins import ensure_components_registered

        cell_line_featurizer_registry.clear()
        ensure_components_registered()
