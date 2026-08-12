"""Which ``_extract_response_pairs`` call sites transform, and which stay raw.

``_extract_response_pairs`` is reached from six places. Four are training-time
supervision and must see transformed targets; two feed prediction and evaluation
and must keep reading the raw response matrix. Getting that split wrong is
silent - the numbers stay plausible, they are just in the wrong space - so the
distinction is pinned here rather than left to review.

The six, as they are exercised below:

===================================================  =========
call site                                            transform
===================================================  =========
``_ComponentStack.train`` output                     yes
``train_with_early_stopping`` output                 yes
``train_with_early_stopping`` early-stopping target  yes
``DRPModel.train`` train_response                    yes
``_ComponentStack.predict`` test_response            no
``DRPModel.predict`` empty-training test_response    no
===================================================  =========

The tests spy on the static method itself instead of asserting on returned
values, because the point is *which* extraction was asked to transform. The spy
forwards the transformer positionally and omits it entirely when it is ``None``,
so the raw-path assertions do not depend on the new parameter existing.

This file is not a module mirror: the contract spans ``component_stack.py`` and
``drp_model.py`` and only means anything when both agree.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler

from drevalpy.models import construct_model
from drevalpy.models.component_stack import _ComponentStack, build_component_stack
from drevalpy.models.config import from_spec
from drevalpy.types import SplitMask, SplitMasks
from drevalpy.utils import fit_response_transformation
from tests.models.synthetic_fixtures import (
    lco_split_masks,
    synthetic_mudataset_gene_expression_fingerprints,
)

ELASTIC_NET_HPAMS = {"alpha": 0.1, "l1_ratio": 0.5}


@pytest.fixture
def mudataset():
    """A two-cell-line, two-drug dataset with gene expression and fingerprints."""
    return synthetic_mudataset_gene_expression_fingerprints()


@pytest.fixture
def masks() -> SplitMasks:
    """LCO-style masks: the first cell line trains, the second is held out."""
    return lco_split_masks()


@pytest.fixture
def fitted(mudataset, masks: SplitMasks) -> TransformerMixin:
    """A ``StandardScaler`` fitted on the training scope, as the pipeline does."""
    return fit_response_transformation(StandardScaler(), mudataset, masks.train)


@pytest.fixture
def extractions(monkeypatch: pytest.MonkeyPatch) -> list[TransformerMixin | None]:
    """Record the transformer handed to every ``_extract_response_pairs`` call."""
    original = _ComponentStack._extract_response_pairs
    recorded: list[TransformerMixin | None] = []

    def recorder(mudataset, scope, response_transformation=None):
        recorded.append(response_transformation)
        if response_transformation is None:
            return original(mudataset, scope)
        return original(mudataset, scope, response_transformation)

    monkeypatch.setattr(_ComponentStack, "_extract_response_pairs", staticmethod(recorder))
    return recorded


class TestTrainingExtractionsAreTransformed:
    def test_the_stack_transforms_its_training_targets(
        self, mudataset, masks: SplitMasks, fitted: TransformerMixin, extractions: list
    ) -> None:
        stack = build_component_stack(from_spec("ElasticNet"))

        stack.train(mudataset, masks.train, response_transformation=fitted)

        assert extractions == [fitted]

    def test_early_stopping_supervision_shares_the_training_space(
        self, mudataset, masks: SplitMasks, fitted: TransformerMixin, extractions: list
    ) -> None:
        """Both extractions here are training-time supervision, so both transform."""
        stack = build_component_stack(from_spec("ElasticNet"))

        stack.train_with_early_stopping(mudataset, masks.train, masks.test, response_transformation=fitted)

        assert extractions == [fitted, fitted]

    def test_the_model_transforms_its_training_targets(
        self, mudataset, masks: SplitMasks, fitted: TransformerMixin, extractions: list
    ) -> None:
        model = construct_model("ElasticNet")(ELASTIC_NET_HPAMS)

        model.train(mudataset=mudataset, scope=masks.train, response_transformation=fitted)

        assert extractions
        assert all(transformer is fitted for transformer in extractions)


class TestEvaluationExtractionsStayRaw:
    def test_the_stack_reads_raw_responses_when_predicting(
        self, mudataset, masks: SplitMasks, fitted: TransformerMixin, extractions: list
    ) -> None:
        stack = build_component_stack(from_spec("ElasticNet"))
        stack.train(mudataset, masks.train, response_transformation=fitted)
        extractions.clear()

        stack.predict(mudataset, masks.test)

        assert extractions == [None]

    def test_the_model_reads_raw_responses_when_predicting(
        self, mudataset, masks: SplitMasks, fitted: TransformerMixin, extractions: list
    ) -> None:
        model = construct_model("ElasticNet")(ELASTIC_NET_HPAMS)
        model.train(mudataset=mudataset, scope=masks.train, response_transformation=fitted)
        extractions.clear()

        model.predict(mudataset=mudataset, scope=masks.test)

        assert extractions == [None]

    def test_the_model_reads_raw_responses_after_empty_training(
        self, mudataset, masks: SplitMasks, fitted: TransformerMixin, extractions: list
    ) -> None:
        """The empty-training shortcut in ``predict`` is the sixth call site."""
        empty = SplitMask(np.zeros(mudataset.response_matrix.shape, dtype=bool))
        model = construct_model("ElasticNet")(ELASTIC_NET_HPAMS)
        model.train(mudataset=mudataset, scope=empty, response_transformation=fitted)
        extractions.clear()

        model.predict(mudataset=mudataset, scope=masks.test)

        assert extractions == [None]


class TestExtractionSemantics:
    """What the transformed extraction actually returns, with the spy out of the way."""

    def test_the_fitted_scaler_is_applied_to_the_responses(
        self, mudataset, masks: SplitMasks, fitted: TransformerMixin
    ) -> None:
        raw = _ComponentStack._extract_response_pairs(mudataset, masks.train)

        transformed = _ComponentStack._extract_response_pairs(mudataset, masks.train, fitted)

        expected = fitted.transform(raw.response.reshape(-1, 1)).ravel()
        np.testing.assert_allclose(transformed.response, expected)

    def test_the_pair_identifiers_are_untouched(self, mudataset, masks: SplitMasks, fitted: TransformerMixin) -> None:
        raw = _ComponentStack._extract_response_pairs(mudataset, masks.train)

        transformed = _ComponentStack._extract_response_pairs(mudataset, masks.train, fitted)

        np.testing.assert_array_equal(transformed.cell_line_ids, raw.cell_line_ids)
        np.testing.assert_array_equal(transformed.drug_ids, raw.drug_ids)

    def test_omitting_the_transform_leaves_the_responses_raw(self, mudataset, masks: SplitMasks) -> None:
        pairs = masks.train.pairs
        expected = mudataset.response_matrix[pairs[:, 0], pairs[:, 1]]

        batch = _ComponentStack._extract_response_pairs(mudataset, masks.train)

        np.testing.assert_allclose(batch.response, expected)

    def test_unmeasured_pairs_are_dropped_before_transforming(self, synthetic_dataset) -> None:
        """The transform runs after the NaN filter, so no NaN reaches the scaler."""
        everything = SplitMask(np.ones(synthetic_dataset.response_matrix.shape, dtype=bool))
        scaler = fit_response_transformation(StandardScaler(), synthetic_dataset, everything)

        batch = _ComponentStack._extract_response_pairs(synthetic_dataset, everything, scaler)

        assert len(batch) == int(np.sum(~np.isnan(synthetic_dataset.response_matrix)))
        assert np.isfinite(batch.response).all()
