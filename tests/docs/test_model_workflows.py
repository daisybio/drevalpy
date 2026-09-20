"""Execute the public API examples exactly as rendered in the model guides."""

from __future__ import annotations

import re
import textwrap

import numpy as np
import pytest
from upath import UPath

from drevalpy import split

pytestmark = pytest.mark.slow
DOCS = UPath(__file__).resolve().parents[2] / "docs"


def _snippets(page: str) -> list[str]:
    text = (DOCS / page).read_text()
    return [
        textwrap.dedent(match)
        for match in re.findall(r"^\.\. code-block:: python\n\n((?: {3}[^\n]*\n|\n)+)", text, re.MULTILINE)
    ]


@pytest.mark.parametrize(
    "page",
    [
        "python/models.rst",
        "project/v2_architecture.rst",
        "project/v2_architecture_migration.rst",
    ],
)
def test_model_guides_use_executable_public_imports(page, synthetic_dataset, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fold = split(synthetic_dataset, "LCO", n_splits=2)[0]
    namespace = {
        "dataset": synthetic_dataset,
        "scope": fold.train,
        "validation_scope": fold.val,
        "fold": fold,
    }
    snippets = _snippets(page)
    assert snippets
    for snippet in snippets:
        exec(compile(snippet, str(DOCS / page), "exec"), namespace)  # noqa: S102 - trusted repository examples
    model = namespace["model"]
    assert np.isfinite(model.predict(synthetic_dataset, fold.test)).any()
    if "restored" in namespace:
        np.testing.assert_allclose(
            namespace["restored"].predict(synthetic_dataset, fold.test),
            model.predict(synthetic_dataset, fold.test),
            equal_nan=True,
        )
    if "results" in namespace:
        assert namespace["results"].n_models == 1
