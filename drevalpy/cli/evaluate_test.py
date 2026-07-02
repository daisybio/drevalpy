"""``drevalpy evaluate-test`` command."""

from __future__ import annotations

from typing import Annotated

import typer

from drevalpy.cli_model_testing import run_evaluate_test_results


def register(app: typer.Typer) -> None:
    @app.command("evaluate-test")
    def evaluate_test(
        model_name: Annotated[str, typer.Option("--model_name", help="Model name.")],
        pred_file: Annotated[str, typer.Option("--pred_file", help="Path to predictions.")],
        test_mode: Annotated[str, typer.Option("--test_mode", help="Test mode (LPO, LCO, LDO, LTO).")] = "LPO",
    ) -> None:
        """Evaluate the predictions."""
        run_evaluate_test_results(test_mode=test_mode, model_name=model_name, pred_file=pred_file)
