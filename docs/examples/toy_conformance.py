"""Testing example: run drevalpy's conformance checks over the toy components.

``drevalpy.testing`` ships the fixtures and checks a plugin's own suite needs, so
no plugin has to hand-roll a dataset or re-derive what "conforms" means. This
module is what the docs build runs to prove the examples on the extensions page
still work.
"""

from __future__ import annotations

import numpy as np

from drevalpy.plugin import ModelResult, RunResult
from drevalpy.registry import splitter
from drevalpy.testing import (
    FEATURIZER_CHECKS,
    PREDICTOR_CHECKS,
    build_synthetic_batch,
    build_synthetic_dataset,
)

from .toy_block_predictor import BLOCK, ToyBlockRidgePredictor
from .toy_cell_line_featurizer import ToyCellLineFeaturizer
from .toy_drug_featurizer import ToyDrugHashFeaturizer
from .toy_mean_predictor import ToyMeanPredictor
from .toy_ridge_predictor import ToyRidgePredictor
from .toy_visualization import ToyResidualHistogram

FEATURIZERS = (ToyCellLineFeaturizer, ToyDrugHashFeaturizer)
PREDICTORS = (ToyMeanPredictor, ToyRidgePredictor, ToyBlockRidgePredictor)


def check_components() -> None:
    """Run every shipped check against every toy featurizer and predictor.

    ``build_synthetic_dataset`` returns a response-only dataset by default; the
    ``omics`` argument adds the cell-line modality ``ToyCellLineFeaturizer``
    reads. ``build_synthetic_batch`` skips featurization entirely and draws the
    feature matrices, which is what makes a predictor testable on its own.
    """
    dataset = build_synthetic_dataset(omics=["gene_expression"])
    batch = build_synthetic_batch(dataset, cell_line_block_names=(BLOCK,))
    for check in FEATURIZER_CHECKS:
        for featurizer in FEATURIZERS:
            check(featurizer, dataset)
    for check in PREDICTOR_CHECKS:
        for predictor in PREDICTORS:
            check(predictor, batch)


def check_splitter() -> None:
    """Run the registered splitter and let the registry validate its folds.

    Resolving the mode through the registry rather than calling the function
    directly is what matters: the registry wrapped it in the ``LCO`` leakage
    check, so a splitter that leaked would raise here.
    """
    dataset = build_synthetic_dataset()
    folds = splitter.get("TOY_LCO")(dataset, n_splits=3)
    if len(folds) != 3:
        msg = f"TOY_LCO produced {len(folds)} folds, expected 3"
        raise AssertionError(msg)


def check_visualization() -> None:
    """Drive the toy visualization through the contract its base class promises.

    There is no conformance check for visualizations yet, so a plot is exercised
    by hand: ``compute`` first, then the renderers, which raise until it has run.
    """
    rng = np.random.default_rng(0)
    run = RunResult(
        model_name="toyRidge",
        dataset_name="SYNTHETIC",
        fold_index=0,
        predictions=rng.normal(size=32),
        ground_truth=rng.normal(size=32),
        cell_line_ids=np.array([f"CVCL_S{index:03d}" for index in range(32)]),
        drug_ids=np.array(["100000"] * 32),
    )
    plot = ToyResidualHistogram()
    plot.compute(ModelResult(model_name="toyRidge", dataset_name="SYNTHETIC", runs=[run]))
    sections = plot.to_multiqc()
    if not sections:
        msg = "ToyResidualHistogram.to_multiqc returned no sections"
        raise AssertionError(msg)
