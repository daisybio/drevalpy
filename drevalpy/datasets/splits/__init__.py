"""Test-mode split providers for drevalpy."""

from .manifest import (
    MANIFEST_FILENAME,
    build_split_manifest,
    read_manifest_test_mode,
    read_split_manifest,
    validate_split_label,
    write_split_manifest,
)
from .providers import (
    create_and_record_splits,
    create_splits,
    load_external_splitter,
    run_builtin_splitter,
    run_external_splitter,
)
from .types import (
    OPTIONAL_ROLES,
    REQUIRED_ROLES,
    TEST_MODES,
    ExternalSplitCreator,
    SplitCreator,
    SplitError,
    SplitFold,
    SplitParams,
    SplitResult,
    make_split_params,
)
from .validation import ensure_early_stopping_splits, validate_splits

__all__ = [
    "MANIFEST_FILENAME",
    "OPTIONAL_ROLES",
    "REQUIRED_ROLES",
    "TEST_MODES",
    "ExternalSplitCreator",
    "SplitCreator",
    "SplitError",
    "SplitFold",
    "SplitParams",
    "SplitResult",
    "build_split_manifest",
    "create_and_record_splits",
    "create_splits",
    "ensure_early_stopping_splits",
    "load_external_splitter",
    "make_split_params",
    "read_manifest_test_mode",
    "read_split_manifest",
    "run_builtin_splitter",
    "run_external_splitter",
    "validate_split_label",
    "validate_splits",
    "write_split_manifest",
]
