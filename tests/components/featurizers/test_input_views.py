"""Every registered featurizer must declare which raw views it reads."""

from __future__ import annotations

import pytest

from drevalpy.components.featurizers._concat import ConcatFeaturizersMixin
from drevalpy.components.registry import (
    get_cell_line_featurizer,
    get_drug_featurizer,
    list_cell_line_featurizers,
    list_drug_featurizers,
)
from drevalpy.components.registry.register_builtins import register_builtin_components

_PROBE_VIEW = "gene_expression"


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


def _featurizer_names(registry: str) -> list[str]:
    register_builtin_components()
    return list_cell_line_featurizers() if registry == "cell_line" else list_drug_featurizers()


@pytest.mark.parametrize("registry", ["cell_line", "drug"])
def test_every_featurizer_declares_input_views(registry: str) -> None:
    get = get_cell_line_featurizer if registry == "cell_line" else get_drug_featurizer
    names = _featurizer_names(registry)
    assert names
    for name in names:
        cls = get(name)
        if issubclass(cls, ConcatFeaturizersMixin):
            continue
        kwargs = {"view": _PROBE_VIEW} if cls.requires_view else {}
        views = cls.resolve_input_views(**kwargs)
        assert isinstance(views, tuple), name
        assert all(isinstance(view, str) and view.strip() for view in views), name


@pytest.mark.parametrize("registry", ["cell_line", "drug"])
def test_entity_id_only_featurizers_need_no_views(registry: str) -> None:
    get = get_cell_line_featurizer if registry == "cell_line" else get_drug_featurizer
    entity_id_only = [name for name in _featurizer_names(registry) if get(name).entity_id_only]
    assert entity_id_only
    for name in entity_id_only:
        assert get(name).resolve_input_views() == (), name


@pytest.mark.parametrize("registry", ["cell_line", "drug"])
def test_concat_featurizer_refuses_standalone_view_resolution(registry: str) -> None:
    get = get_cell_line_featurizer if registry == "cell_line" else get_drug_featurizer
    cls = get("concatFeaturizers")
    with pytest.raises(TypeError, match="has no input views of its own"):
        cls.resolve_input_views()


@pytest.mark.parametrize("name", ["raw", "pca"])
def test_view_parameterized_featurizers_require_an_explicit_view(name: str) -> None:
    cls = get_cell_line_featurizer(name)
    assert cls.requires_view
    assert cls.resolve_input_views(view="mutations") == ("mutations",)
    with pytest.raises(TypeError, match="requires an explicit view"):
        cls.resolve_input_views()
