"""End-to-end ``DRPModel`` runs: zoo presets, train/predict/checkpoint round trips and best-epoch reloads."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from drevalpy.models import DRPModel, ModelSpec
from drevalpy.models.config import ModelConfig, from_dict, from_spec, validate
from drevalpy.models.factory import model_config_for_name
from drevalpy.models.zoo import list_zoo_names
from drevalpy.types.data.split_mask import SplitMask
from tests._trusted_subprocess import run_trusted_python
from tests.models.synthetic_fixtures import (
    lso_split_masks,
    small_data_spec,
    synthetic_model_dataset,
    synthetic_model_dataset_gene_expression_fingerprints,
    synthetic_model_dataset_identity,
)


@pytest.mark.parametrize("name", list_zoo_names(include_external=False))
def test_zoo_entries_resolve_and_build_models(name: str) -> None:
    config = model_config_for_name(name)
    assert isinstance(config, ModelConfig)
    validate(config)
    assert config.predictor.name
    assert isinstance(from_spec(name), ModelConfig)
    assert DRPModel(ModelSpec.from_name(name)).name == name


def test_no_pair_context_in_production_code() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "drevalpy"
    hits = [p for p in root.rglob("*.py") if "pair_context" in p.read_text(encoding="utf-8")]
    assert not hits


_NAIVE_PRESETS = (
    "NaivePredictor",
    "NaiveDrugMeanPredictor",
    "NaiveSampleMeanPredictor",
)
_SKLEARN_PRESETS = (
    "ElasticNet",
    "RandomForest",
    "Lasso",
    "SVR",
    "SingleDrugElasticNet",
    "SingleDrugRandomForest",
)


@pytest.mark.parametrize(
    ("preset", "naive"),
    [(preset, True) for preset in _NAIVE_PRESETS] + [(preset, False) for preset in _SKLEARN_PRESETS],
)
def test_direct_component_round_trip(preset: str, naive: bool, tmp_path) -> None:
    dataset = synthetic_model_dataset_identity() if naive else synthetic_model_dataset_gene_expression_fingerprints()
    split = lso_split_masks()
    model = DRPModel(ModelSpec.from_name(preset) if naive else small_data_spec(preset))
    model.train(dataset, split.train)
    preds = model.predict(dataset, split.test)
    assert preds.shape[0] > 0
    assert np.isfinite(preds).all()
    model.save(tmp_path / "model")
    assert np.allclose(preds, DRPModel.load(tmp_path / "model").predict(dataset, split.test))


def test_featurizer_requirements_by_predictor_scope() -> None:
    with pytest.raises(ValueError, match="requires a drug_featurizer"):
        validate(from_dict({"sample_featurizer": "scaledGeneExpression", "predictor": "elasticNet"}))

    single = from_dict({"sample_featurizer": "scaledGeneExpression", "predictor": "singleDrugElasticNet"})
    validate(single)
    assert single.drug_featurizer is not None
    assert single.drug_featurizer.name == "identity"

    validate(from_dict({"predictor": "naiveMean"}))


@pytest.mark.slow
def test_subprocess_blocks_optional_deps_for_simple_models() -> None:
    """Spawns an interpreter because the pristine import graph is the assertion."""
    script = textwrap.dedent("""
        import importlib.abc
        import importlib.machinery
        import sys

        blocked = {
            "xgboost": "blocked xgboost",
            "lightgbm": "blocked lightgbm",
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
                if fullname in blocked or fullname.split(".", 1)[0] in blocked:
                    key = fullname if fullname in blocked else fullname.split(".", 1)[0]
                    return importlib.machinery.ModuleSpec(fullname, BlockLoader(blocked[key]))
                return None

        sys.meta_path.insert(0, BlockFinder())

        import anndata as ad
        import mudata as md
        import numpy as np
        import pandas as pd

        from drevalpy.types.data.dataset import Dataset
        from drevalpy.types import SplitMask, SplitMasks
        from drevalpy.models import DRPModel, ModelSpec
        from drevalpy.models.config import from_spec

        cl_ids = np.array(["cl1", "cl2"])
        drug_ids = np.array(["d1", "d2"])
        response_matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        response_ad = ad.AnnData(
            X=response_matrix,
            obs=pd.DataFrame({"cell_line_name": cl_ids, "tissue": ["L", "B"]}, index=cl_ids),
            var=pd.DataFrame(index=drug_ids),
        )
        ge_matrix = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
        ge_ad = ad.AnnData(
            X=ge_matrix,
            obs=pd.DataFrame(index=cl_ids),
            var=pd.DataFrame(index=["g0", "g1", "g2"]),
        )
        response_ad.varm["morgan_fingerprint"] = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        mdata = md.MuData({"response": response_ad, "gene_expression": ge_ad})
        dataset_ge = Dataset(mdata, name="test")

        response_ad2 = ad.AnnData(
            X=response_matrix.copy(),
            obs=pd.DataFrame({"cell_line_name": cl_ids, "tissue": ["L", "B"]}, index=cl_ids),
            var=pd.DataFrame(index=drug_ids),
        )
        mdata2 = md.MuData({"response": response_ad2})
        dataset_id = Dataset(mdata2, name="test")

        split = SplitMasks(
            train=SplitMask(np.array([[True, True], [False, False]])),
            test=SplitMask(np.array([[False, False], [True, True]])),
            val=SplitMask(np.zeros((2, 2), dtype=bool)),
        )

        naive = DRPModel(ModelSpec.from_name("NaivePredictor"), {})
        naive.train(dataset_id, split.train)
        elastic = DRPModel(ModelSpec("ElasticNet", from_spec("scaledGeneExpression:fingerprints:elasticNet")))
        elastic.train(dataset_ge, split.train)
        """)
    completed = run_trusted_python(script)
    assert completed.returncode == 0, completed.stdout + completed.stderr


RF_HP = {"n_estimators": 8, "max_depth": 3, "max_samples": 1.0, "random_state": 0, "n_jobs": 1}


@pytest.mark.parametrize(
    ("model_name", "hp"),
    [
        ("NaiveDrugMeanPredictor", {}),
        ("ElasticNet", {"alpha": 0.1, "l1_ratio": 0.5}),
        ("RandomForest", RF_HP),
    ],
)
def test_equivalent_specs_have_matching_predictions(model_name: str, hp: dict) -> None:
    naive = model_name.startswith("Naive")
    spec = ModelSpec.from_name(model_name) if naive else small_data_spec(model_name)
    dataset = synthetic_model_dataset_identity() if naive else synthetic_model_dataset_gene_expression_fingerprints()
    split = lso_split_masks()
    first, second = DRPModel(spec, hp), DRPModel(ModelSpec(model_name, spec.config), hp)
    first.train(dataset, split.train)
    second.train(dataset, split.train)
    np.testing.assert_allclose(first.predict(dataset, split.test), second.predict(dataset, split.test), equal_nan=True)


def test_custom_recipe_spec_has_matching_checkpoint_round_trip(tmp_path) -> None:
    spec = ModelSpec("PcaIdentityRF", from_spec("pca[gene_expression]:identity:randomForest"))
    dataset = synthetic_model_dataset_gene_expression_fingerprints()
    split = lso_split_masks()
    model = DRPModel(spec, RF_HP)
    model.train(dataset, split.train)
    model.save(tmp_path / "model")
    loaded = DRPModel.load(tmp_path / "model")
    np.testing.assert_allclose(model.predict(dataset, split.test), loaded.predict(dataset, split.test), equal_nan=True)
    assert loaded.name == "PcaIdentityRF"


_SMALL_HYPERPARAMETERS = {
    "DIPK": {"epochs": 1, "epochs_autoencoder": 1, "heads": 1},
    "SimpleNeuralNetwork": {"units_per_layer": [2, 2], "max_epochs": 1},
    "MultiViewNeuralNetwork": {"units_per_layer": [2, 2], "max_epochs": 1},
    "PharmaFormer": {"epochs": 1, "patience": 2},
    "Precily": {"epochs": 1, "batch_size": 32},
    "SparseGO": {"epochs": 1, "batch_size": 32},
}


@pytest.mark.parametrize(
    "model_name",
    [
        "DIPK",
        "DrugGNN",
        "MOLIR",
        "PharmaFormer",
        "Precily",
        "SRMF",
        "SimpleNeuralNetwork",
        "MultiViewNeuralNetwork",
        "SuperFELTR",
        "SparseGO",
    ],
)
def test_literature_models_build_with_defaults(model_name: str) -> None:
    spec = ModelSpec.from_name(model_name)
    DRPModel(spec, {**spec.default_hyperparameters, **_SMALL_HYPERPARAMETERS.get(model_name, {})})


@pytest.mark.parametrize(
    ("model_name", "hyperparameters", "extra_views"),
    [
        ("SimpleNeuralNetwork", {"units_per_layer": [2, 2], "max_epochs": 1}, ()),
        ("SRMF", {"K": 2, "max_iter": 2, "n_features": 4}, ()),
        ("MultiViewNeuralNetwork", {"units_per_layer": [2, 2], "max_epochs": 1}, ("mutations",)),
    ],
)
def test_literature_model_lifecycle(
    model_name: str,
    hyperparameters: dict,
    extra_views: tuple[str, ...],
    tmp_path,
) -> None:
    dataset = synthetic_model_dataset(n_features_per_view=4, fingerprint_width=4, extra_views=extra_views)
    split = lso_split_masks()
    model = DRPModel(small_data_spec(model_name), hyperparameters)
    model.train(dataset, split.train)
    preds = model.predict(dataset, split.test)
    assert preds.shape[0] > 0
    assert np.isfinite(preds).all()

    model.save(tmp_path / "model")
    loaded_preds = DRPModel.load(tmp_path / "model").predict(dataset, split.test)
    assert np.allclose(preds, loaded_preds, rtol=1e-5, atol=1e-5)


def _three_omics_dataset():
    import anndata as ad
    import mudata as md
    import pandas as pd

    from drevalpy.types.data.dataset import Dataset

    rng = np.random.default_rng(0)
    sample_ids = np.array([f"cl{i}" for i in range(8)])
    response = ad.AnnData(
        X=rng.normal(size=(8, 1)).astype(np.float32),
        obs=pd.DataFrame({"cell_line_name": sample_ids, "tissue": ["Lung", "Blood"] * 4}, index=sample_ids),
        var=pd.DataFrame(index=["d1"]),
    )
    response.varm["morgan_fingerprint"] = np.ones((1, 2), dtype=np.float32)
    modalities = {"response": response}
    for view, prefix in (("gene_expression", "g"), ("mutations", "m"), ("copy_number_variation_gistic", "c")):
        modalities[view] = ad.AnnData(
            X=rng.normal(size=(8, 4)).astype(np.float32),
            obs=pd.DataFrame(index=sample_ids),
            var=pd.DataFrame(index=[f"{prefix}{i}" for i in range(4)]),
        )
    return Dataset(md.MuData(modalities), name="test")


def _spy_on_best_epoch_reload(monkeypatch: pytest.MonkeyPatch, model_name: str) -> list[object]:
    reloaded: list[object] = []
    if model_name == "MOLIR":
        from drevalpy.components.predictors.literature.molir import utils

        original = utils.load_state_dict

        def spy(path, *args, **kwargs):
            reloaded.append(original(path, *args, **kwargs))
            return reloaded[-1]

        monkeypatch.setattr(utils, "load_state_dict", spy)
    else:
        from drevalpy.components.predictors.literature.superfeltr.utils import SuperFELTEncoder, SuperFELTRegressor

        for network in (SuperFELTEncoder, SuperFELTRegressor):
            original = network.load_from_checkpoint.__func__

            def spy(cls, *args, _original=original, **kwargs):
                reloaded.append(_original(cls, *args, **kwargs))
                return reloaded[-1]

            monkeypatch.setattr(network, "load_from_checkpoint", classmethod(spy))
    return reloaded


@pytest.mark.slow
@pytest.mark.parametrize("model_name", ["MOLIR", "SuperFELTR"])
def test_fit_with_early_stopping_scope_reloads_the_best_epoch(
    model_name: str, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    hyperparameters = {"epochs": 2, "mini_batch": 2, "dropout_rate": 0.0}
    if model_name == "MOLIR":
        hyperparameters.update({"h_dim1": 2, "h_dim2": 2, "h_dim3": 2})
    else:
        hyperparameters.update({"out_dim_expr_encoder": 2, "out_dim_mutation_encoder": 2, "out_dim_cnv_encoder": 2})
    mask = np.zeros((8, 1), dtype=bool)
    train, stop = mask.copy(), mask.copy()
    train[:6] = True
    stop[6:] = True

    reloaded = _spy_on_best_epoch_reload(monkeypatch, model_name)
    dataset = _three_omics_dataset()
    model = DRPModel(small_data_spec(model_name), hyperparameters)
    model.train(dataset, SplitMask(train), early_stopping_scope=SplitMask(stop), model_checkpoint_dir=tmp_path)

    assert reloaded, "the fit never reached the best-epoch reload"
    assert np.isfinite(model.predict(dataset, SplitMask(stop))).all()
