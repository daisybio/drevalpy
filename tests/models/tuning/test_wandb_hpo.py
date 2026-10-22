"""W&B tracking of Optuna HPO: one run per trial."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from drevalpy.models import ModelSpec
from drevalpy.models.tuning import hpo
from drevalpy.models.tuning.config import HPOConfig
from drevalpy.models.tuning.hpo import HPOTrialsFailedError, hpam_tune
from drevalpy.types import SplitMask
from tests._fake_wandb import FakeWandb, install_fake_wandb
from tests._import_shims import block_imports
from tests.models.synthetic_fixtures import synthetic_mudataset_gene_expression_fingerprints

pytestmark = pytest.mark.slow


def _tune(**kwargs):
    mudataset = synthetic_mudataset_gene_expression_fingerprints()
    pairs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    scope = SplitMask.from_pairs(pairs, shape=mudataset.response_matrix.shape)
    return hpam_tune(
        model_spec=ModelSpec.from_name("ElasticNet"),
        mudataset=mudataset,
        train_scope=scope,
        val_scope=scope,
        early_stopping_scope=None,
        metric="RMSE",
        hpo_config=HPOConfig.from_metric("RMSE", n_trials=2),
        **kwargs,
    )


@pytest.fixture
def fake_wandb(monkeypatch: pytest.MonkeyPatch) -> FakeWandb:
    return install_fake_wandb(monkeypatch)


def test_each_trial_is_one_named_run_with_its_metrics_in_the_summary(fake_wandb: FakeWandb) -> None:
    _tune(split_index=3, wandb_project="proj", wandb_base_config={"dataset": "synthetic"})

    assert fake_wandb.run_names() == ["ElasticNet_split3_trial0", "ElasticNet_split3_trial1"]
    for run in fake_wandb.runs:
        assert run.project == "proj"
        assert run.finished
        assert run.exit_code is None
        assert run.config["dataset"] == "synthetic"
        assert len(run.config) > 1
        assert run.summary["hpo_metric"] == run.summary["val_RMSE"]
        assert {"val_R^2", "val_Pearson", "val_RMSE"} <= set(run.summary)


def test_without_a_project_no_run_is_started(fake_wandb: FakeWandb) -> None:
    _tune(split_index=0)

    assert fake_wandb.runs == []


def test_a_failing_trial_leaves_a_failed_run(fake_wandb: FakeWandb, monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(hpo, "_mu_evaluate_trial_all_metrics", explode)

    with pytest.raises(HPOTrialsFailedError):
        _tune(split_index=0, wandb_project="proj")

    assert [run.exit_code for run in fake_wandb.runs] == [1, 1]


def test_a_missing_wandb_fails_before_the_first_trial(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    block_imports(monkeypatch, "wandb")

    with pytest.raises(ImportError, match="pip install wandb"):
        _tune(split_index=0, wandb_project="proj")
