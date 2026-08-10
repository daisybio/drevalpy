"""Integration test: full pipeline load → split → train → predict."""

from __future__ import annotations

import pytest

from drevalpy.data import load, split
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
            assert fold.train.mask.ndim == 2
            assert fold.test.mask.ndim == 2
            assert fold.train.mask.dtype == bool
            assert fold.train.any()
            assert fold.test.any()

    def test_elastic_net_train_predict(self, mudataset):
        """ElasticNet should train and predict via the run() function without errors."""
        from drevalpy.data.splitters import splitter_registry
        from drevalpy.single import single

        ElasticNet = construct_model("ElasticNet")  # noqa: N806

        splitter = splitter_registry.get("LCO")
        folds = splitter(mudataset, n_splits=2, validation_ratio=0.2)

        result = single(
            model_class=ElasticNet,
            mudataset=mudataset,
            split_masks=folds[0],
            hyperparameter_tuning=False,
        )
        assert len(result.predictions) > 0
        assert result.metrics
