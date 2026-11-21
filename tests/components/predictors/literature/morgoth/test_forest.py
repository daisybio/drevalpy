"""Tests of the sklearn wrapper of the MORGOTH forest (``literature/morgoth/_forest.py``)."""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas
import pytest
from sklearn.base import clone

from drevalpy.components.predictors.literature.morgoth import _fast_split
from drevalpy.components.predictors.literature.morgoth._forest import MorgothRegressor, _pool, import_morgoth


def _data(n: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Draw a small regression problem.

    :param n: Number of samples.
    :param seed: Random seed.
    :returns: Feature matrix and responses.
    """
    rng = np.random.RandomState(seed)
    x = rng.normal(size=(n, 4))
    return x, x[:, 0] + rng.normal(scale=0.3, size=n)


def _morgoth_module() -> types.ModuleType:
    """Return ``morgoth.morgoth`` imported behind the fireducks alias.

    :returns: The module ``morgoth.morgoth``.
    """
    import_morgoth()
    return sys.modules["morgoth.morgoth"]


def test_import_morgoth_returns_the_forest_class() -> None:
    """The returned class is the package's MORGOTH."""
    assert import_morgoth() is _morgoth_module().MORGOTH


def test_missing_morgoth_raises_the_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the package, the error names the extra to install."""
    monkeypatch.setitem(sys.modules, "morgoth.morgoth", None)
    with pytest.raises(ImportError, match=r"drevalpy\[morgoth\]") as info:
        import_morgoth()
    assert isinstance(info.value.__cause__, ImportError)


def test_alias_fireducks_aliases_pandas_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without fireducks, ``fireducks.pandas`` becomes pandas."""

    def missing(name: str) -> None:
        raise ImportError(name)

    monkeypatch.delitem(sys.modules, "fireducks", raising=False)
    monkeypatch.delitem(sys.modules, "fireducks.pandas", raising=False)
    monkeypatch.setattr(_fast_split, "importlib", types.SimpleNamespace(import_module=missing))
    _fast_split.alias_fireducks()
    assert sys.modules["fireducks.pandas"] is pandas
    assert sys.modules["fireducks"].pandas is pandas


def test_alias_fireducks_keeps_an_installed_fireducks(monkeypatch: pytest.MonkeyPatch) -> None:
    """An importable fireducks is not overridden."""
    fake_pandas = types.ModuleType("fireducks.pandas")
    fake = types.ModuleType("fireducks")
    fake.pandas = fake_pandas
    monkeypatch.setitem(sys.modules, "fireducks", fake)
    monkeypatch.setitem(sys.modules, "fireducks.pandas", fake_pandas)
    _fast_split.alias_fireducks()
    assert sys.modules["fireducks.pandas"] is fake_pandas
    assert sys.modules["fireducks"] is fake


def test_pool_swaps_and_restores_the_package_pool() -> None:
    """``_pool`` replaces ``morgoth.morgoth.Pool`` inside the block only, also after an error."""
    module = _morgoth_module()
    original = module.Pool
    with _pool(1):
        assert module.Pool is not original
        with module.Pool() as pool:
            assert pool.map(abs, [-1, 2]) == [1, 2]
    assert module.Pool is original
    with pytest.raises(ZeroDivisionError), _pool(2):
        _ = 1 / 0
    assert module.Pool is original


def test_clone_keeps_the_parameters() -> None:
    """``sklearn.base.clone`` copies the parameters of an unfitted regressor."""
    regressor = MorgothRegressor(n_estimators=3, max_features=0.5, fast_split=False)
    assert clone(regressor).get_params() == regressor.get_params()


class TestFit:
    """Fits of tiny single-process forests."""

    @staticmethod
    def _regressor(**kwargs: object) -> MorgothRegressor:
        r"""Build a tiny single-process regressor.

        :param \**kwargs: Parameters overriding the tiny defaults.
        :returns: The unfitted regressor.
        """
        params = {"n_estimators": 2, "max_depth": 3, "min_samples_leaf": 5, "n_jobs": 1} | kwargs
        return MorgothRegressor(**params)

    def test_predict_returns_floats_per_sample(self) -> None:
        """The prediction has one float per row of X."""
        x, y = _data()
        prediction = self._regressor().fit(x, y).predict(x[:7])
        assert prediction.dtype == float
        assert prediction.shape == (7,)

    def test_without_classes_the_threshold_is_the_quantile(self) -> None:
        """Without ``y_class`` the threshold is the ``threshold_quantile`` of the responses."""
        x, y = _data()
        regressor = self._regressor(threshold_quantile=0.3).fit(x, y)
        assert regressor.threshold_ == pytest.approx(np.quantile(y, 0.3))

    def test_low_sensitive_responses_use_the_class_fraction(self) -> None:
        """For ln-IC50-like responses (class 1 = low), the quantile is the fraction of class 1."""
        x, y = _data()
        y_class = (y < np.quantile(y, 0.25)).astype(int)
        regressor = self._regressor().fit(x, y, y_class=y_class)
        assert regressor.threshold_ == pytest.approx(np.quantile(y, y_class.mean()))

    def test_high_sensitive_responses_flip_the_quantile(self) -> None:
        """For pEC50-like responses (class 1 = high), the quantile is one minus the fraction of class 1."""
        x, y = _data()
        y_class = (y > np.quantile(y, 0.75)).astype(int)
        regressor = self._regressor().fit(x, y, y_class=y_class)
        assert regressor.threshold_ == pytest.approx(np.quantile(y, 1 - y_class.mean()))

    def test_a_single_class_fits(self) -> None:
        """Class labels with one class only still fit; the threshold is the extreme response."""
        x, y = _data()
        regressor = self._regressor().fit(x, y, y_class=np.zeros(len(y), dtype=int))
        assert regressor.threshold_ == pytest.approx(y.min())
        assert np.isfinite(regressor.predict(x)).all()

    def test_fit_leaves_no_files_in_the_working_directory(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The files the package writes go to a temporary directory.

        :param tmp_path: Empty working directory.
        :param monkeypatch: Changes into ``tmp_path``.
        """
        monkeypatch.chdir(tmp_path)
        x, y = _data()
        self._regressor().fit(x, y).predict(x)
        assert list(tmp_path.iterdir()) == []
