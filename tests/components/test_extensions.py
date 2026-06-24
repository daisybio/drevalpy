"""Tests for external component and zoo loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drevalpy.components.config import ModelConfig
from drevalpy.components.extensions import load_extension_dir, load_extension_file, load_extensions
from drevalpy.components.registry import get_cell_line_featurizer, get_predictor, list_cell_line_featurizers
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.zoo import get_zoo_config, list_zoo_names, load_external_zoo_file


def test_load_extension_file_registers_components(tmp_path: Path) -> None:
    ext_file = tmp_path / "toy_extension.py"
    ext_file.write_text(
        """
from __future__ import annotations

import numpy as np

from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.predictors.baseline import BaselinePredictor
from drevalpy.components.registry import register_cell_line_featurizer, register_predictor


@register_cell_line_featurizer("toyCellLine", description="Toy featurizer", category="general_purpose")
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
    "toyPredictor", description="Toy predictor", category="general_purpose")
class ToyPredictor(BaselinePredictor):
    def fit(self, x, y, *, pair_context=None):
        return None

    def predict(self, x, *, pair_context=None):
        return np.zeros(len(x), dtype=np.float64)
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
        "from drevalpy.components.predictors.baseline import BaselinePredictor\n"
        "import numpy as np\n"
        "@register_predictor('toyB', description='b', category='general_purpose')\n"
        "class ToyB(BaselinePredictor):\n"
        "    def fit(self, x, y, *, pair_context=None): return None\n"
        "    def predict(self, x, *, pair_context=None): return np.zeros(len(x))\n",
        encoding="utf-8",
    )
    (tmp_path / "a_ext.py").write_text(
        "from drevalpy.components.registry import register_predictor\n"
        "from drevalpy.components.predictors.baseline import BaselinePredictor\n"
        "import numpy as np\n"
        "@register_predictor('toyA', description='a', category='general_purpose')\n"
        "class ToyA(BaselinePredictor):\n"
        "    def fit(self, x, y, *, pair_context=None): return None\n"
        "    def predict(self, x, *, pair_context=None): return np.zeros(len(x))\n",
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
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.predictors.baseline import BaselinePredictor
from drevalpy.components.registry import register_cell_line_featurizer, register_predictor

@register_cell_line_featurizer("externalCellLine", description="ext", category="general_purpose")
class ExternalCellLineFeaturizer(CellLineFeaturizer):
    def fit(self, features, *, entity_ids=None):
        self._output_dim = 1
        return self
    def transform(self, features, entity_ids):
        return np.ones((len(entity_ids), 1), dtype=np.float32)
    @property
    def output_dim(self):
        return self._output_dim

@register_predictor("externalPredictor", description="ext", category="general_purpose")
class ExternalPredictor(BaselinePredictor):
    def fit(self, x, y, *, pair_context=None):
        self._mean = float(np.mean(y))
    def predict(self, x, *, pair_context=None):
        return np.full(len(x), self._mean, dtype=np.float64)
""",
        encoding="utf-8",
    )
    zoo_file = tmp_path / "external_zoo.yaml"
    zoo_file.write_text(
        """
externalToy:
  cell_line_featurizer:
    type: externalCellLine
  predictor:
    type: externalPredictor
""",
        encoding="utf-8",
    )
    load_extensions(directories=[ext_dir], zoo_files=[zoo_file])
    assert "externalToy" in list_zoo_names(include_external=True)
    config = get_zoo_config("externalToy")
    model = config.create_model()
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
predictor:
  type: naiveMean
""",
        encoding="utf-8",
    )
    names = load_external_zoo_file(zoo_file)
    assert names == ["customNaive"]
    config = get_zoo_config("customNaive")
    assert isinstance(config, ModelConfig)
    assert config.predictor.type == "naiveMean"


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
