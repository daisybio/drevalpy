"""Tests for checkpoint directory resolution."""

from __future__ import annotations

from pathlib import Path

from drevalpy.utils.checkpoints import checkpoint_dir_or_temporary, resolve_checkpoint_dir


def test_none_requests_a_temporary_directory() -> None:
    """``None`` defers the choice to the caller's temporary-directory fallback."""
    assert resolve_checkpoint_dir(None) is None


def test_legacy_temporary_string_is_still_accepted() -> None:
    """The retired ``"TEMPORARY"`` sentinel keeps working for existing command lines."""
    assert resolve_checkpoint_dir("TEMPORARY") is None


def test_string_directory_becomes_a_path() -> None:
    """A plain string is normalized to a ``Path``."""
    assert resolve_checkpoint_dir("checkpoints") == Path("checkpoints")


def test_path_directory_is_preserved() -> None:
    """An explicit ``Path`` is passed through unchanged."""
    explicit = Path("relative") / "ckpt"
    assert resolve_checkpoint_dir(explicit) == explicit


def test_directory_literally_named_temporary_is_kept_as_a_path() -> None:
    """``Path("TEMPORARY")`` is a real directory request, not the sentinel."""
    assert resolve_checkpoint_dir(Path("TEMPORARY")) == Path("TEMPORARY")


def test_context_manager_creates_a_temporary_directory_for_none() -> None:
    """``None`` yields a real directory that is cleaned up on exit."""
    with checkpoint_dir_or_temporary(None) as checkpoint_dir:
        assert checkpoint_dir.is_dir()
        created = checkpoint_dir
    assert not created.exists()


def test_context_manager_creates_a_requested_directory(tmp_path: Path) -> None:
    """A requested directory is created, including missing parents, and kept.

    :param tmp_path: Temporary directory used as the checkpoint root.
    """
    requested = tmp_path / "nested" / "checkpoints"
    with checkpoint_dir_or_temporary(requested) as checkpoint_dir:
        assert checkpoint_dir == requested
        assert checkpoint_dir.is_dir()
    assert requested.is_dir()


def test_context_manager_always_yields_an_existing_directory(tmp_path: Path) -> None:
    """A ``str`` argument behaves exactly like the equivalent ``Path``.

    :param tmp_path: Temporary directory used as the checkpoint root.
    """
    with checkpoint_dir_or_temporary(str(tmp_path / "from_str")) as checkpoint_dir:
        assert checkpoint_dir == tmp_path / "from_str"
        assert checkpoint_dir.is_dir()
