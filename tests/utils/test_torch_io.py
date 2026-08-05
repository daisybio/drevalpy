"""Tests for the trusted PyTorch serialization boundary."""

from __future__ import annotations

import io
import pickle
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from drevalpy.components.predictors.literature._torch_state import (
    load_object_mapping,
)
from drevalpy.components.predictors.literature._torch_state import load_state_dict as load_state_dict_bytes
from drevalpy.components.predictors.literature._torch_state import (
    save_object_mapping,
)
from drevalpy.components.predictors.literature._torch_state import save_state_dict as save_state_dict_bytes
from drevalpy.utils.torch_io import (
    load_state_dict,
    load_torch_payload,
    load_trusted_mapping,
    load_trusted_payload,
    save_torch_payload,
    save_trusted_mapping,
)


def test_state_dict_bytes_round_trip() -> None:
    state = {"layer.weight": torch.tensor([1.0, 2.0])}
    loaded = load_state_dict_bytes(save_state_dict_bytes(state))
    assert torch.equal(loaded["layer.weight"], state["layer.weight"])


def test_trusted_mapping_bytes_round_trip() -> None:
    payload = {"hyperparameters": {"epochs": 3}, "value": torch.tensor(4.0)}
    loaded = load_object_mapping(save_object_mapping(payload))
    assert loaded["hyperparameters"] == {"epochs": 3}
    value = loaded["value"]
    expected = payload["value"]
    assert isinstance(value, torch.Tensor)
    assert isinstance(expected, torch.Tensor)
    assert torch.equal(value, expected)


def test_state_dict_rejects_non_mapping_payload() -> None:
    buffer = io.BytesIO()
    save_torch_payload(torch.tensor([1.0, 2.0]), buffer)
    with pytest.raises(TypeError, match="state dict mapping"):
        load_state_dict(buffer.getvalue())


def test_trusted_mapping_rejects_non_mapping_payload() -> None:
    buffer = io.BytesIO()
    save_torch_payload(torch.tensor([1.0, 2.0]), buffer)
    with pytest.raises(TypeError, match="mapping"):
        load_trusted_mapping(buffer.getvalue())


def test_load_state_dict_from_path(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.pt"
    state = {"layer.bias": torch.tensor([0.5])}
    save_torch_payload(state, checkpoint_path)
    loaded = load_state_dict(checkpoint_path)
    assert torch.equal(loaded["layer.bias"], state["layer.bias"])


def test_load_state_dict_honors_map_location(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.pt"
    state = {"layer.weight": torch.tensor([1.0], device="cpu")}
    save_torch_payload(state, checkpoint_path)
    loaded = load_state_dict(checkpoint_path, map_location="cpu")
    assert loaded["layer.weight"].device.type == "cpu"


def test_load_trusted_payload_restores_graph_objects(tmp_path: Path) -> None:
    graph = Data(
        x=torch.ones((2, 3)),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
    )
    graph_path = tmp_path / "graph.pt"
    save_torch_payload(graph, graph_path)
    loaded = load_trusted_payload(graph_path)
    assert isinstance(loaded, Data)
    assert torch.equal(loaded.x, graph.x)
    assert torch.equal(loaded.edge_index, graph.edge_index)


def test_legacy_trusted_checkpoint_compatibility() -> None:
    payload = {
        "hyperparameters": {"max_epochs": 1},
        "input_dim": 4,
        "state_dict": {"net.weight": torch.randn(2, 4)},
    }
    restored = load_trusted_mapping(save_trusted_mapping(payload))
    assert restored["input_dim"] == 4
    assert restored["hyperparameters"] == {"max_epochs": 1}
    assert "net.weight" in restored["state_dict"]


def test_load_torch_payload_rejects_invalid_bytes() -> None:
    with pytest.raises(pickle.UnpicklingError):
        load_torch_payload(b"not-a-torch-checkpoint")
