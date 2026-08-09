"""test mu_hpam_tune with Ray Tune."""

import importlib.util

import pytest

from drevalpy import experiment
from drevalpy.components.tuning.config import HPOConfig
from drevalpy.models import construct_model


def test_hpam_tune(tmp_path, data_dir):
    """Test mu_hpam_tune with a toy MuDataset and ElasticNet model.

    :param tmp_path: pytest temporary path fixture
    :param data_dir: path to the data directory
    """
    if importlib.util.find_spec("ray") is None:
        pytest.skip("Ray is not installed")

    from drevalpy.data import load_mudataset
    from drevalpy.data.splitting import MuDataSplitter
    from drevalpy.experiment.fold import prepare_mu_fold

    model_cls = construct_model("ElasticNet")
    mudataset = load_mudataset("TOYv1")
    splitter = MuDataSplitter()
    folds = splitter.split(mudataset, mode="LPO", n_splits=2, validation_ratio=0.4)
    split = folds[0]
    fold_data = prepare_mu_fold(mudataset, split, model_cls)

    best = experiment.mu_hpam_tune(
        model_class=model_cls,
        mudataset=mudataset,
        train_scope=fold_data.train_scope,
        val_scope=fold_data.val_scope,
        early_stopping_scope=fold_data.early_stopping_scope,
        metric="RMSE",
        hpo_config=HPOConfig.from_metric("RMSE", n_trials=2, storage_path=str(tmp_path)),
    )
    assert isinstance(best, dict)
    assert "alpha" in best
