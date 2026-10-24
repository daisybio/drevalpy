"""Tests for drevalpy.data._paths cache directory resolution."""

from pathlib import Path

import pytest
from platformdirs import user_cache_dir

from drevalpy.data._paths import get_default_data_dir


@pytest.mark.parametrize("pad", ["", "  "])
def test_env_var_override_is_stripped(monkeypatch, tmp_path, pad):
    custom = str(tmp_path / "custom_cache")
    monkeypatch.setenv("DREVALPY_CACHE_DIR", f"{pad}{custom}{pad}")
    assert get_default_data_dir() == Path(custom)


@pytest.mark.parametrize("value", ["", None])
def test_empty_or_unset_env_var_uses_platformdirs(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("DREVALPY_CACHE_DIR", raising=False)
    else:
        monkeypatch.setenv("DREVALPY_CACHE_DIR", value)
    assert get_default_data_dir() == Path(user_cache_dir("drevalpy"))
