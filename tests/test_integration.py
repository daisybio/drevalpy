"""Integration test: full pipeline load → split → train → predict."""

from __future__ import annotations

import pytest

from drevalpy.data import load, split
from drevalpy.experiment import mu_experiment
from drevalpy.models import construct_model


@pytest.fixture
def mudataset():
    """Load a small dataset for integration testing."""
    try:
        return load("CTRPv1")
    except (FileNotFoundError, OSError):
        pytest.skip("CTRPv1 dataset not available in cache")


class TestFullPipeline:
    def test_split_produces_valid_folds(self, mudataset):
        folds = split(mudataset, "LCO", n_splits=2)
        assert len(folds) == 2
        for fold in folds:
            assert fold.train.shape[1] == 2
            assert fold.test.shape[1] == 2
            assert len(fold.train) > 0
            assert len(fold.test) > 0

    def test_elastic_net_train_predict(self, mudataset):
        """ElasticNet should train and predict without NaN errors."""
        from drevalpy.data.splitters import splitter_registry
        from drevalpy.data.structures import EntityScope
        from drevalpy.experiment.fold import prepare_mu_fold

        ElasticNet = construct_model("ElasticNet")

        splitter = splitter_registry.get("LCO")
        folds = splitter(mudataset, n_splits=2, validation_ratio=0.2)
        fold_data = prepare_mu_fold(mudataset, folds[0], ElasticNet)

        model = ElasticNet()
        model.train(
            mudataset=mudataset,
            train_scope=fold_data.train_scope,
            early_stopping_scope=fold_data.early_stopping_scope,
        )

        predictions = model.predict(
            mudataset=mudataset,
            test_scope=fold_data.test_scope,
        )
        assert len(predictions) > 0
        assert not any(predictions != predictions)  # no NaN in predictions
