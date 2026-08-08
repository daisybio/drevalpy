"""Tests for drevalpy.datasets._paths cache directory resolution."""

from pathlib import Path

from drevalpy.datasets._paths import get_default_data_dir


def test_env_var_override(monkeypatch, tmp_path):
    """DREVALPY_CACHE_DIR should be used verbatim when set."""
    custom = str(tmp_path / "custom_cache")
    monkeypatch.setenv("DREVALPY_CACHE_DIR", custom)
    assert get_default_data_dir() == Path(custom)


def test_env_var_whitespace_stripped(monkeypatch, tmp_path):
    """Surrounding whitespace in the env var should be stripped."""
    custom = str(tmp_path / "custom_cache")
    monkeypatch.setenv("DREVALPY_CACHE_DIR", f"  {custom}  ")
    assert get_default_data_dir() == Path(custom)


def test_empty_env_var_uses_platformdirs(monkeypatch):
    """An empty (but set) env var should fall back to platformdirs."""
    monkeypatch.setenv("DREVALPY_CACHE_DIR", "")
    from platformdirs import user_cache_dir

    assert get_default_data_dir() == Path(user_cache_dir("drevalpy"))


def test_unset_env_var_uses_platformdirs(monkeypatch):
    """An unset env var should fall back to platformdirs."""
    monkeypatch.delenv("DREVALPY_CACHE_DIR", raising=False)
    from platformdirs import user_cache_dir

    assert get_default_data_dir() == Path(user_cache_dir("drevalpy"))
