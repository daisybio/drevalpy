"""CLI/experiment paths must not emit legacy factory or view FutureWarnings."""

from __future__ import annotations

import pathlib
import warnings

from drevalpy.models import construct_model
from drevalpy.models._model_lookup import get_model_class, known_model_names
from drevalpy.utils import check_arguments
from drevalpy.utils._deprecations import reset_deprecation_warnings


def _dreval_future_warnings(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [
        w
        for w in caught
        if issubclass(w.category, FutureWarning)
        and (
            "MODEL_FACTORY" in str(w.message)
            or "MULTI_DRUG_MODEL_FACTORY" in str(w.message)
            or "SINGLE_DRUG_MODEL_FACTORY" in str(w.message)
            or "Legacy cell_line_views/drug_views" in str(w.message)
        )
    ]


def test_check_arguments_and_model_lookup_do_not_warn(tmp_path: pathlib.Path) -> None:
    reset_deprecation_warnings()

    class _Args:
        models = ["ElasticNet"]
        baselines = ["NaivePredictor"]
        test_mode = ["LPO"]
        dataset_name = "GDSC1"
        no_refitting = True
        curve_curator_cores = 1
        cross_study_datasets: list[str] = []
        n_cv_splits = 2
        custom_splitter_path = None
        custom_split_name = None
        randomization_mode = ["None"]
        randomization_type = "permutation"
        n_trials_robustness = 0
        response_transformation = "None"
        measure = "LN_IC50"
        optim_metric = "Pearson"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert "ElasticNet" in known_model_names(include_external=False)
        assert get_model_class("ElasticNet").get_model_name() == "ElasticNet"
        assert construct_model("ElasticNet").get_model_name() == "ElasticNet"
        check_arguments(_Args())

    assert _dreval_future_warnings(caught) == []
