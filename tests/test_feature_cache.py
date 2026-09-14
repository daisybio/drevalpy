"""Tests for the feature matrix cache used by the experiment pipeline."""

import ast
import inspect
import textwrap

import numpy as np
import pytest

from drevalpy import experiment
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.experiment import _load_features_cached, clear_feature_cache, load_features
from drevalpy.models import MODEL_FACTORY
from drevalpy.models.drp_model import DRPModel

#: Number of loader calls per feature kind, shared by all test models so the cache effect is visible
#: across instances and subclasses.
_CALLS = {"cell_line": 0, "drug": 0}


class _CountingModel(DRPModel):
    """Model whose loaders count how often they were called and return a fresh object each time."""

    cell_line_views = ["gene_expression"]
    drug_views = ["fingerprints"]

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: name of this test model
        """
        return "CountingModel"

    @classmethod
    def get_hyperparameter_set(cls) -> list[dict]:
        """
        Returns a single empty hyperparameter set, there is no hyperparameters.yaml for a test model.

        :returns: list with one empty hyperparameter dict
        """
        return [{}]

    def build_model(self, hyperparameters: dict) -> None:
        """
        Stores the hyperparameters, which are part of the cache key.

        :param hyperparameters: hyperparameters to use
        """
        self.hyperparameters = hyperparameters

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Returns a new FeatureDataset and counts the call.

        :param data_path: unused
        :param dataset_name: unused
        :returns: cell line features
        """
        _CALLS["cell_line"] += 1
        return FeatureDataset(features={"CL0": {"gene_expression": np.zeros(3)}})

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset | None:
        """
        Returns a new FeatureDataset and counts the call.

        :param data_path: unused
        :param dataset_name: unused
        :returns: drug features
        """
        _CALLS["drug"] += 1
        return FeatureDataset(features={"D0": {"fingerprints": np.zeros(3)}})

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """
        Not used by these tests.

        :param output: unused
        :param cell_line_input: unused
        :param drug_input: unused
        :param output_earlystopping: unused
        :param model_checkpoint_dir: unused
        """

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Returns zeros.

        :param cell_line_ids: cell line ids to predict for
        :param drug_ids: unused
        :param cell_line_input: unused
        :param drug_input: unused
        :returns: zero predictions
        """
        return np.zeros(len(cell_line_ids))


class _OtherCountingModel(_CountingModel):
    """Same behaviour under a different class name, so it must get its own cache entry."""

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: name of this test model
        """
        return "OtherCountingModel"


class _StatefulLoaderModel(_CountingModel):
    """Model whose loader also initializes model state, like SparseGO builds its ontology there."""

    supports_feature_caching = False

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: name of this test model
        """
        return "StatefulLoaderModel"

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Returns features and, as a side effect, initializes state that train would need.

        :param data_path: unused
        :param dataset_name: unused
        :returns: cell line features
        """
        features = super().load_cell_line_features(data_path=data_path, dataset_name=dataset_name)
        self.ontology_loaded = True
        return features


def _self_assignments(function) -> set[str]:
    """
    Collect the attribute names a function assigns on ``self``.

    :param function: function to inspect
    :returns: names assigned on self, empty if the function only returns something
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    assigned = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign | ast.AugAssign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                    if target.value.id == "self":
                        assigned.add(target.attr)
    return assigned


@pytest.fixture(autouse=True)
def _reset_cache():
    """
    Start every test with an empty cache and zeroed call counters.

    :yields: nothing, this fixture only manages global state
    """
    clear_feature_cache()
    _CALLS["cell_line"] = 0
    _CALLS["drug"] = 0
    yield
    clear_feature_cache()


def _build(hyperparameters: dict | None = None, model_class: type[_CountingModel] = _CountingModel) -> _CountingModel:
    """
    Instantiate and build a counting model.

    :param hyperparameters: hyperparameters to build with
    :param model_class: class to instantiate
    :returns: the built model
    """
    model = model_class()
    model.build_model(hyperparameters=hyperparameters if hyperparameters is not None else {})
    return model


def test_repeated_loads_hit_the_cache() -> None:
    """The same model, path and dataset must load each feature kind exactly once."""
    model = _build()
    first = _load_features_cached(model, "data", "TOYv1", "cell_line")
    second = _load_features_cached(model, "data", "TOYv1", "cell_line")

    assert _CALLS["cell_line"] == 1
    assert first is second


def test_cache_is_shared_across_model_instances() -> None:
    """A fresh instance with the same hyperparameters must reuse the cached matrix.

    This is the case that matters in practice: single drug models re-instantiate the model
    for every drug and every split.
    """
    for _ in range(5):
        _load_features_cached(_build(), "data", "TOYv1", "cell_line")

    assert _CALLS["cell_line"] == 1


def test_cache_key_separates_kind_dataset_path_and_hyperparameters() -> None:
    """Everything the matrices depend on has to be part of the cache key."""
    model = _build()
    _load_features_cached(model, "data", "TOYv1", "cell_line")
    _load_features_cached(model, "data", "TOYv1", "drug")
    assert (_CALLS["cell_line"], _CALLS["drug"]) == (1, 1)

    _load_features_cached(model, "data", "TOYv2", "cell_line")
    assert _CALLS["cell_line"] == 2

    _load_features_cached(model, "other_data", "TOYv1", "cell_line")
    assert _CALLS["cell_line"] == 3

    _load_features_cached(_build({"gene_list": "landmark_genes_reduced"}), "data", "TOYv1", "cell_line")
    assert _CALLS["cell_line"] == 4

    _load_features_cached(_build(model_class=_OtherCountingModel), "data", "TOYv1", "cell_line")
    assert _CALLS["cell_line"] == 5


def test_hyperparameter_order_does_not_split_the_key() -> None:
    """Equal hyperparameters written in a different order describe the same matrix."""
    _load_features_cached(_build({"a": 1, "b": 2}), "data", "TOYv1", "cell_line")
    _load_features_cached(_build({"b": 2, "a": 1}), "data", "TOYv1", "cell_line")

    assert _CALLS["cell_line"] == 1


def test_cache_evicts_oldest_entry_at_maxsize() -> None:
    """The cache must not grow without bound, an entry is a full feature matrix."""
    for i in range(experiment._FEATURE_CACHE_MAXSIZE + 1):
        _load_features_cached(_build(), "data", f"TOY{i}", "cell_line")

    assert len(experiment._FEATURE_CACHE) == experiment._FEATURE_CACHE_MAXSIZE
    assert _CALLS["cell_line"] == experiment._FEATURE_CACHE_MAXSIZE + 1

    # The first dataset was evicted and has to be loaded again.
    _load_features_cached(_build(), "data", "TOY0", "cell_line")
    assert _CALLS["cell_line"] == experiment._FEATURE_CACHE_MAXSIZE + 2


def test_cache_can_be_disabled_via_environment(monkeypatch) -> None:
    """DREVAL_FEATURE_CACHE=0 restores the previous load-every-time behaviour.

    :param monkeypatch: pytest monkeypatch fixture
    """
    monkeypatch.setenv("DREVAL_FEATURE_CACHE", "0")
    model = _build()
    first = _load_features_cached(model, "data", "TOYv1", "cell_line")
    second = _load_features_cached(model, "data", "TOYv1", "cell_line")

    assert _CALLS["cell_line"] == 2
    assert first is not second
    assert experiment._FEATURE_CACHE == {}


def test_load_features_uses_the_cache() -> None:
    """load_features is the entry point used by the pipeline and has to go through the cache."""
    dataset = DrugResponseDataset(
        response=np.array([1.0]),
        cell_line_ids=np.array(["CL0"]),
        drug_ids=np.array(["D0"]),
        dataset_name="TOYv1",
    )
    model = _build()
    cl_first, drug_first = load_features(model, "data", dataset)
    cl_second, drug_second = load_features(model, "data", dataset)

    assert (_CALLS["cell_line"], _CALLS["drug"]) == (1, 1)
    assert cl_first is cl_second
    assert drug_first is drug_second


def test_models_that_opt_out_load_for_every_instance() -> None:
    """A loader that initializes model state has to run for every instance, not once per process."""
    models = [_build(model_class=_StatefulLoaderModel) for _ in range(3)]
    for model in models:
        _load_features_cached(model, "data", "TOYv1", "cell_line")

    assert _CALLS["cell_line"] == 3
    assert experiment._FEATURE_CACHE == {}
    assert all(getattr(model, "ontology_loaded", False) for model in models)


def test_stateful_loaders_are_marked_as_uncacheable() -> None:
    """Every shipped model whose feature loaders assign to self must set supports_feature_caching False.

    The cache hands the same matrix to later instances and skips their loader, so a loader that also
    initializes model state would leave those instances half built.
    """
    stateful = {
        name
        for name, model_class in MODEL_FACTORY.items()
        if _self_assignments(model_class.load_cell_line_features) or _self_assignments(model_class.load_drug_features)
    }
    not_opted_out = {name for name in stateful if MODEL_FACTORY[name].supports_feature_caching}

    assert not_opted_out == set(), (
        f"{sorted(not_opted_out)} initialize model state in their feature loaders and must set "
        "supports_feature_caching = False"
    )
