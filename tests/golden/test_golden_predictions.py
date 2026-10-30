"""Golden gate: every zoo model reproduces its recorded predictions and metrics bit for bit.

The baseline in ``tests/golden/data/`` was recorded by ``python -m tests.golden._record`` on the
unmodified code; see that module for the fold, the seeds and how to adapt the one training helper.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from tests.golden._record import (
    BASELINE_FILE,
    hyperparameter_values,
    metrics_equal,
    predictions_match,
    round_trip_predict,
    train_and_predict,
)

pytestmark = pytest.mark.slow


def _params() -> list:
    with BASELINE_FILE.open(encoding="utf-8") as handle:
        baseline = json.load(handle)
    recorded = [
        pytest.param(key, id=key, marks=[pytest.mark.network] if entry["artifacts"] else [])
        for key, entry in sorted(baseline["models"].items())
    ]
    network = [pytest.param(key, id=key, marks=pytest.mark.network) for key in sorted(baseline["network_only"])]
    return recorded + network


@pytest.mark.parametrize("key", _params())
def test_golden_predictions(key, golden_dataset, golden_fold, golden_baseline, golden_predictions) -> None:
    """Predictions, metrics and a save-load-predict round trip match the baseline."""
    if key in golden_baseline["network_only"]:
        pytest.skip(f"no offline baseline: {golden_baseline['network_only'][key]}")
    entry = golden_baseline["models"][key]
    run = train_and_predict(
        entry["model"],
        golden_dataset,
        golden_fold,
        hyperparameter_tuning=entry["hyperparameter_tuning"],
        overrides=entry["hyperparameter_overrides"],
    )

    expected = golden_predictions[key]
    assert predictions_match(run.predictions, expected, exact=entry["deterministic"]), (run.predictions, expected)
    if entry["deterministic"]:
        assert metrics_equal(run.metrics, entry["metrics"]), (run.metrics, entry["metrics"])
    else:
        assert run.metrics.keys() == entry["metrics"].keys()
        names = sorted(entry["metrics"])
        np.testing.assert_allclose(
            [run.metrics[name] for name in names], [entry["metrics"][name] for name in names], rtol=0, atol=1e-5
        )
    if entry["hyperparameter_tuning"]:
        chosen = Counter(hyperparameter_values(run.best_hyperparameters))
        missing = Counter(entry["best_hyperparameter_values"]) - chosen
        assert not missing, f"tuned values {sorted(missing)} not chosen any more: {run.best_hyperparameters}"

    loaded = round_trip_predict(run, golden_dataset, golden_fold)
    assert predictions_match(loaded, run.predictions, exact=entry["round_trip_exact"])
