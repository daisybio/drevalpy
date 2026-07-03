"""Tests for legacy ``drevalpy-*`` console script aliases."""

from __future__ import annotations

import warnings

import pytest

from drevalpy.cli.legacy import load_response, train_and_predict_cv


def test_legacy_alias_emits_deprecation_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["drevalpy-train-cv", "--help"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(SystemExit):
            train_and_predict_cv()

    assert any(issubclass(w.category, FutureWarning) and "drevalpy train-cv" in str(w.message) for w in caught)


def test_legacy_alias_invalid_option_exits_without_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid legacy alias flags should exit via Click, not bubble Python exceptions."""
    monkeypatch.setattr("sys.argv", ["drevalpy-load-response", "--not-a-real-flag"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with pytest.raises(SystemExit) as exc_info:
            load_response()

    assert exc_info.value.code != 0
