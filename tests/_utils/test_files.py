"""Portable file names and trusted torch payload io."""

from __future__ import annotations

import io
import pickle
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from drevalpy._utils.filenames import safe_filename, unique_safe_filenames
from drevalpy._utils.torch_io import (
    load_state_dict,
    load_torch_payload,
    load_trusted_mapping,
    load_trusted_payload,
    save_state_dict,
    save_torch_payload,
    save_trusted_mapping,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ElasticNet", "ElasticNet"),
        ("pca[gene_expression]:fingerprints:ridge", "pca[gene_expression]_fingerprints_ridge"),
        ('a<b>c"d/e\\f|g?h*i', "a_b_c_d_e_f_g_h_i"),
        ("tab\there", "tab_here"),
        ("trailing. ", "trailing"),
        ("...", "_"),
    ],
)
def test_safe_filename(name: str, expected: str) -> None:
    assert safe_filename(name) == expected


def test_unique_safe_filenames_suffixes_collisions_case_insensitively() -> None:
    assert unique_safe_filenames(["a:b", "a_b", "A_B", "c"]) == {"a:b": "a_b", "a_b": "a_b-2", "A_B": "A_B-3", "c": "c"}


def test_bytes_round_trips() -> None:
    state = {"layer.weight": torch.tensor([1.0, 2.0])}
    assert torch.equal(load_state_dict(save_state_dict(state))["layer.weight"], state["layer.weight"])

    payload = {"hyperparameters": {"epochs": 3}, "value": torch.tensor(4.0)}
    loaded = load_trusted_mapping(save_trusted_mapping(payload))
    assert loaded["hyperparameters"] == {"epochs": 3}
    assert torch.equal(loaded["value"], payload["value"])


@pytest.mark.parametrize(
    ("loader", "match"), [(load_state_dict, "state dict mapping"), (load_trusted_mapping, "mapping")]
)
def test_non_mapping_payload_is_rejected(loader, match) -> None:
    buffer = io.BytesIO()
    save_torch_payload(torch.tensor([1.0, 2.0]), buffer)
    with pytest.raises(TypeError, match=match):
        loader(buffer.getvalue())


@pytest.mark.parametrize("as_str", [False, True])
def test_load_state_dict_from_path_and_map_location(tmp_path: Path, as_str: bool) -> None:
    path = tmp_path / "model.pt"
    state = {"layer.bias": torch.tensor([0.5])}
    save_torch_payload(state, path)
    loaded = load_state_dict(str(path) if as_str else path, map_location="cpu")
    assert torch.equal(loaded["layer.bias"], state["layer.bias"])
    assert loaded["layer.bias"].device.type == "cpu"


def test_load_trusted_payload_restores_graph_objects(tmp_path: Path) -> None:
    graph = Data(x=torch.ones((2, 3)), edge_index=torch.tensor([[0], [1]], dtype=torch.long))
    save_torch_payload(graph, tmp_path / "graph.pt")
    loaded = load_trusted_payload(tmp_path / "graph.pt")
    assert isinstance(loaded, Data)
    assert torch.equal(loaded.x, graph.x)
    assert torch.equal(loaded.edge_index, graph.edge_index)


def test_load_torch_payload_rejects_invalid_bytes() -> None:
    with pytest.raises((pickle.UnpicklingError, RuntimeError)):
        load_torch_payload(b"not-a-torch-checkpoint")
