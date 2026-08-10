"""Compatibility tests for drevalpy.datasets.custom_splits."""

from __future__ import annotations

from pathlib import Path

from drevalpy.datasets import custom_splits
from drevalpy.datasets.custom_splits import (
    CustomSplitError,
    CustomSplitParams,
    SplitError,
    SplitParams,
    load_custom_splitter,
    run_splitter,
)
from tests.datasets.split_helpers import sample_dataset


def test_custom_splits_exports_aliases() -> None:
    """Legacy custom_splits names remain available as aliases."""
    assert CustomSplitError is SplitError
    assert CustomSplitParams is SplitParams
    assert custom_splits.load_custom_splitter is custom_splits.load_external_splitter
    assert custom_splits.validate_cv_splits is custom_splits.validate_splits


def test_load_custom_splitter_alias(tmp_path: Path) -> None:
    """
    Legacy loader alias resolves external split scripts.

    :param tmp_path: Temporary path provided by pytest.
    """
    script = tmp_path / "splitter.py"
    script.write_text(
        """
def create_splits(response_data, params):
    return []
""",
        encoding="utf-8",
    )
    assert callable(load_custom_splitter(script))


def test_run_splitter_alias_delegates_to_create_splits() -> None:
    """Compatibility alias still creates built-in splits."""
    dataset = sample_dataset(n_cell_lines=4, n_drugs=2)
    splits, metadata = run_splitter(
        dataset,
        test_mode="LPO",
        n_cv_splits=2,
        validation_ratio=0.2,
        random_state=11,
        split_early_stopping=False,
    )
    assert len(splits) == 2
    assert metadata[0]["split_index"] == 0
