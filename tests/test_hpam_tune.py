"""Test hpam_tune with Optuna against the synthetic dataset."""

from drevalpy.models import construct_model
from drevalpy.models.tuning.config import HPOConfig
from drevalpy.models.tuning.hpo import hpam_tune
from drevalpy.types.data.dataset import Dataset


def test_hpam_tune(synthetic_dataset: Dataset):
    """Tune ElasticNet over a Leave-Pairs-Out fold of the synthetic dataset.

    :param synthetic_dataset: Session-scoped synthetic raw-omics dataset.
    """
    from drevalpy.registry.splitter import get as get_splitter

    model_cls = construct_model("ElasticNet")
    splitter = get_splitter("LPO")
    folds = splitter(synthetic_dataset, n_splits=2, validation_ratio=0.4)
    split = folds[0]

    early_stopping_scope = None
    val_scope = split.val
    if model_cls.supports_early_stopping() and len(split.val) > 1:
        early_stopping_scope, val_scope = split.early_stopping_mask()

    best, _ = hpam_tune(
        model_class=model_cls,
        mudataset=synthetic_dataset,
        train_scope=split.train,
        val_scope=val_scope,
        early_stopping_scope=early_stopping_scope,
        metric="RMSE",
        hpo_config=HPOConfig.from_metric("RMSE", n_trials=2),
    )
    assert isinstance(best, dict)
    assert "alpha" in best
