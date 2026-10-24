"""Normalization of a run against a reference model's run, used by :meth:`ExperimentResult.normalize`."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from drevalpy._grouped_evaluation import run_group_score
from drevalpy.evaluation import score_predictions
from drevalpy.types.results.run import RunResult


def normalize_run(run: RunResult, ref_run: RunResult) -> RunResult:
    """Normalize a single RunResult against a reference RunResult.

    :param run: Run to normalize.
    :param ref_run: Ordinary run of the reference model on the same fold.
    :returns: A copy of ``run`` with reference-subtracted predictions and ground truth, recomputed metrics, and the
        unnormalized pooled metrics and per-drug Pearson kept in ``raw_metrics`` and ``raw_per_drug_pearson``.
    """
    import pandas as pd

    ref_index = pd.MultiIndex.from_arrays(
        [np.asarray(ref_run.sample_ids, dtype=object), np.asarray(ref_run.drug_ids, dtype=object)]
    )
    ref_predictions = np.asarray(ref_run.predictions)
    if ref_index.has_duplicates:
        # The dict lookup this replaced let the last occurrence of a pair win.
        keep = ~ref_index.duplicated(keep="last")
        ref_index = ref_index[keep]
        ref_predictions = ref_predictions[keep]
    positions = ref_index.get_indexer(
        pd.MultiIndex.from_arrays([np.asarray(run.sample_ids, dtype=object), np.asarray(run.drug_ids, dtype=object)])
    )
    # get_indexer yields -1 for pairs the reference never predicted; those normalize
    # against 0.0, matching the dict-lookup default this replaced.
    ref_preds = np.where(positions >= 0, ref_predictions[positions.clip(min=0)], 0.0)

    norm_gt = run.ground_truth - ref_preds
    norm_pred = run.predictions - ref_preds

    return replace(
        run,
        predictions=norm_pred,
        ground_truth=norm_gt,
        metrics=score_predictions(norm_pred, norm_gt, empty_as_dict=True),
        trials=[replace(trial) for trial in run.trials] if run.trials else None,
        raw_metrics=dict(run.metrics),
        raw_per_drug_pearson=run_group_score(run, "drug", "Pearson"),
    )


def reference_summary(ref_run: RunResult) -> RunResult:
    """Summarize a reference run without its arrays, as the leaderboard needs it.

    The reference model is normalized against itself, so its normalized predictions are all zero. The
    arrays are dropped because only the metrics are read, and a report must not keep them in memory.

    :param ref_run: Ordinary run of the reference model.
    :returns: The run normalized against itself, with empty prediction and identifier arrays.
    """
    empty = np.empty(0)
    return replace(
        normalize_run(ref_run, ref_run),
        predictions=empty,
        ground_truth=empty,
        sample_ids=np.empty(0, dtype=object),
        drug_ids=np.empty(0, dtype=object),
        trials=None,
    )
