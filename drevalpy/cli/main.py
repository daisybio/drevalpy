"""Typer entry point for the ``drevalpy`` console script."""

from __future__ import annotations

import sys

import typer

from drevalpy.cli import (
    collect_results,
    consolidate_single_drug,
    evaluate_hpams,
    evaluate_test,
    load_response,
    make_cv_pkls,
    make_final_split_pkls,
    make_hpam_yamls,
    make_pipeline_report,
    make_randomization_yamls,
    pipeline,
    report,
    test_cv,
    train_cv,
    train_final_model_cmd,
    tune_final_model,
    viability_postprocess,
    viability_preprocess,
)
from drevalpy.cli._helpers import normalize_list_argv

app = typer.Typer(
    name="drevalpy",
    help="Drug response evaluation of cancer cell line drug response models in a fair setting.",
    no_args_is_help=False,
)

pipeline.register_pipeline_callback(app)
viability_preprocess.register(app)
viability_postprocess.register(app)
load_response.register(app)
make_cv_pkls.register(app)
make_hpam_yamls.register(app)
train_cv.register(app)
evaluate_hpams.register(app)
test_cv.register(app)
make_randomization_yamls.register(app)
make_final_split_pkls.register(app)
tune_final_model.register(app)
train_final_model_cmd.register(app)
consolidate_single_drug.register(app)
evaluate_test.register(app)
collect_results.register(app)
report.register(app)
make_pipeline_report.register(app)


def cli_main() -> None:
    """Poetry console script entry point."""
    app(normalize_list_argv(sys.argv[1:]))


if __name__ == "__main__":
    cli_main()
