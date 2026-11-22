"""Shared batch builders for predictor unit tests."""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from drevalpy.types.data.feature_block import FeatureBlock, FeaturizerOutput
from drevalpy.types.data.predictor_input import PredictorInput
from drevalpy.types.data.response_table import ResponseTable
from drevalpy.types.data.training_context import TrainingContext
from tests.models.synthetic_fixtures import (
    drug_fingerprints,
    multi_drug_response,
    sample_gene_expression,
)

if TYPE_CHECKING:
    import pytest

_T = TypeVar("_T")


def pickle_round_trip(obj: _T) -> _T:
    """Return *obj* pickled and loaded again, the way a fitted predictor is persisted.

    :param obj: Object to round-trip.
    :returns: The loaded copy.
    """
    return pickle.loads(pickle.dumps(obj))  # noqa: S301 -- pickled one line earlier


def neural_batch(*, with_early_stopping: bool = False) -> PredictorInput:
    """Build a dense matrix batch for predictors that train on flattened features.

    :param with_early_stopping: Attach a validation :class:`ResponseTable` when true.
    :returns: Featurized ``PredictorInput`` with a temporary checkpoint directory.
    """
    response = multi_drug_response()
    sample_features = np.vstack(
        [
            sample_gene_expression().features["cl1"]["gene_expression"],
            sample_gene_expression().features["cl2"]["gene_expression"],
        ]
    )
    drug_features = np.vstack(
        [
            drug_fingerprints().features["d1"]["fingerprints"],
            drug_fingerprints().features["d2"]["fingerprints"],
        ]
    )
    early_stopping = None
    if with_early_stopping:
        early_stopping = ResponseTable(
            response=np.array([1.5, 2.5]),
            sample_ids=np.array(["cl1", "cl2"]),
            drug_ids=np.array(["d1", "d2"]),
        )
    return PredictorInput(
        response,
        sample=FeaturizerOutput(np.array(["cl1", "cl2"]), {"gene_expression": FeatureBlock(sample_features)}),
        drug=FeaturizerOutput(np.array(["d1", "d2"]), {"fingerprints": FeatureBlock(drug_features)}),
        early_stopping=early_stopping,
        training_context=TrainingContext(checkpoint_dir=Path(tempfile.mkdtemp())),
    )


def spy_trainer(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Record the keyword arguments of every ``pytorch_lightning.Trainer`` built from now on.

    :param monkeypatch: The test's monkeypatch fixture; the patch is undone with it.
    :returns: A list that gains one dict of ``Trainer`` keyword arguments per construction.
    """
    import pytorch_lightning as pl

    seen: list[dict[str, object]] = []
    real_trainer = pl.Trainer

    def spy(*args, **kwargs):
        seen.append(dict(kwargs))
        return real_trainer(*args, **kwargs)

    monkeypatch.setattr(pl, "Trainer", spy)
    return seen
