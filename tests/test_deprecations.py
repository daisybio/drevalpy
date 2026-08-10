"""Tests for shared deprecation helpers."""

from __future__ import annotations

import warnings

from drevalpy.utils._deprecations import reset_deprecation_warnings, warn_deprecated


def test_warn_deprecated_emits_future_warning_once() -> None:
    reset_deprecation_warnings()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_deprecated(what="legacy.api", replacement="modern.api()", stacklevel=2)
        warn_deprecated(what="legacy.api", replacement="modern.api()", stacklevel=2)

    future = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert len(future) == 1
    assert "legacy.api" in str(future[0].message)
    assert "modern.api()" in str(future[0].message)


def test_reset_allows_warning_again() -> None:
    reset_deprecation_warnings()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_deprecated(what="legacy.reset", replacement="modern()", stacklevel=2)
    assert any(issubclass(w.category, FutureWarning) for w in caught)

    reset_deprecation_warnings()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_deprecated(what="legacy.reset", replacement="modern()", stacklevel=2)
    assert any(issubclass(w.category, FutureWarning) for w in caught)
