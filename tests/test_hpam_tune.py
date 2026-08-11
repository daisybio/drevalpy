"""Test hpam_tune with Optuna."""

from drevalpy.models import construct_model
from drevalpy.models.tuning.config import HPOConfig
from drevalpy.models.tuning.hpo import hpam_tune


def test_hpam_tune(tmp_path, data_dir):
    """Test hpam_tune with a toy Dataset and ElasticNet model.

    :param tmp_path: pytest temporary path fixture
    :param data_dir: path to the data directory
    """
    from drevalpy.data import load
    from drevalpy.data.splitters import get_splitter

    model_cls = construct_model("ElasticNet")
    mudataset = load("TOYv1")
    splitter = get_splitter("LPO")
    folds = splitter(mudataset, n_splits=2, validation_ratio=0.4)
    split = folds[0]

    early_stopping_scope = None
    val_scope = split.val
    if model_cls.supports_early_stopping() and len(split.val) > 1:
        early_stopping_scope, val_scope = split.early_stopping_mask()

    best, _ = hpam_tune(
        model_class=model_cls,
        mudataset=mudataset,
        train_scope=split.train,
        val_scope=val_scope,
        early_stopping_scope=early_stopping_scope,
        metric="RMSE",
        hpo_config=HPOConfig.from_metric("RMSE", n_trials=2),
    )
    assert isinstance(best, dict)
    assert "alpha" in best
