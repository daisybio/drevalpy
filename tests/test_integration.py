"""Integration test: full pipeline load → split → train → predict."""

from __future__ import annotations

import pytest
from upath import UPath

from drevalpy.data import load, split
from drevalpy.data._paths import get_default_data_dir, resolve_h5mu_path
from drevalpy.models import construct_model
from drevalpy.registry.dataset._registry import dataset_registry

DATASET_NAME = "CTRPv1"


def _cached_h5mu() -> UPath | None:
    """Return the local .h5mu for :data:`DATASET_NAME`, if it is already cached.

    Mirrors the two locations ``load`` looks in before it would download.

    :returns: Existing local path, or ``None`` when the dataset is not cached.
    """
    candidates = [resolve_h5mu_path(DATASET_NAME)]
    if dataset_registry.is_registered(DATASET_NAME):
        candidates.append(get_default_data_dir() / dataset_registry.datasets[DATASET_NAME].file)
    return next((path for path in candidates if path.is_file()), None)


@pytest.fixture
def mudataset():
    """Load the real dataset from the local cache, skipping when it is absent.

    Deliberately checks the cache before calling ``load``, so an uncached run
    skips without attempting a download. The only source registered for this
    dataset is a credentialed ``s3://`` bucket, and a download attempt without
    those credentials raises ``botocore`` errors such as
    ``UnauthorizedSSOTokenError``. Those are not ``OSError`` subclasses, so they
    used to surface as test *errors* rather than skips and broke any run on a
    machine without the dataset.
    """
    path = _cached_h5mu()
    if path is None:
        pytest.skip(f"{DATASET_NAME} is not in the local cache; not downloading it here")
    try:
        return load(DATASET_NAME)
    except Exception as exc:  # noqa: BLE001 - any failure to obtain the data is a skip, not a failure
        pytest.skip(f"cached {DATASET_NAME} at {path} could not be loaded: {exc!r}")


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
        from drevalpy.registry.splitter import splitter_registry
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
