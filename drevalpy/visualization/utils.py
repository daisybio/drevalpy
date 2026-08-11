"""Backward-compatible re-export (moved to _legacy)."""

from drevalpy.visualization._legacy import utils as _utils


# Re-export public API via __getattr__ for full backward compatibility
def __getattr__(name: str):  # noqa: N807
    return getattr(_utils, name)
