"""Tests for the drug featurizer validation shim.

``drevalpy/registry/drug_featurizer/_validate.py`` is a pure re-export of the shared
featurizer validation, so the only behaviour to pin is the identity of the
re-exported name. The validation itself is covered in
``tests/registry/featurizer/test_validate.py``.
"""

from __future__ import annotations

from drevalpy.registry.drug_featurizer import _validate
from drevalpy.registry.featurizer._validate import validate_featurizer_input_views


def test_re_exports_the_shared_validator() -> None:
    assert _validate.validate_featurizer_input_views is validate_featurizer_input_views


def test_exports_only_the_shared_validator() -> None:
    assert _validate.__all__ == ["validate_featurizer_input_views"]
