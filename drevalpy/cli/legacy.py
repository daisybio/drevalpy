"""Legacy ``drevalpy-*`` console scripts as aliases to Typer subcommands."""

from __future__ import annotations

import sys
from collections.abc import Callable

from drevalpy.cli._helpers import normalize_list_argv
from drevalpy.cli._legacy import warn_deprecated


def _legacy_alias(legacy_script: str, subcommand: str) -> Callable[[], None]:
    """Return a Poetry entry point that forwards to ``drevalpy <subcommand>``."""

    def entrypoint() -> None:
        warn_deprecated(legacy_script=legacy_script, replacement=f"drevalpy {subcommand}")
        from drevalpy.cli.main import app

        app(normalize_list_argv([subcommand, *sys.argv[1:]]), prog_name=legacy_script)

    entrypoint.__doc__ = f"Legacy alias for ``drevalpy {subcommand}``."
    return entrypoint


preprocess_raw_viability = _legacy_alias("drevalpy-viability-preprocess", "viability-preprocess")
postprocess_viability = _legacy_alias("drevalpy-viability-postprocess", "viability-postprocess")
load_response = _legacy_alias("drevalpy-load-response", "load-response")
cv_split = _legacy_alias("drevalpy-make-cv-pkls", "make-cv-pkls")
hpam_split = _legacy_alias("drevalpy-make-hpam-yamls", "make-hpam-yamls")
train_and_predict_cv = _legacy_alias("drevalpy-train-cv", "train-cv")
evaluate_and_find_max = _legacy_alias("drevalpy-evaluate-hpams", "evaluate-hpams")
train_and_predict_final = _legacy_alias("drevalpy-test-cv", "test-cv")
randomization_split = _legacy_alias("drevalpy-make-randomization-yamls", "make-randomization-yamls")
final_split = _legacy_alias("drevalpy-make-final-split-pkls", "make-final-split-pkls")
tune_final_model = _legacy_alias("drevalpy-tune-final-model", "tune-final-model")
train_final_model = _legacy_alias("drevalpy-train-final-model", "train-final-model")
consolidate_results = _legacy_alias("drevalpy-consolidate-single-drug", "consolidate-single-drug")
evaluate_test_results = _legacy_alias("drevalpy-evaluate-test", "evaluate-test")
collect_results = _legacy_alias("drevalpy-collect-results", "collect-results")
main = _legacy_alias("drevalpy-report", "report")
pipeline_report = _legacy_alias("drevalpy-make-pipeline-report", "make-pipeline-report")
