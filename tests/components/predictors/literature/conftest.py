"""Process-global isolation for seeded literature predictor tests."""

from __future__ import annotations

import random
from collections.abc import Iterator

import numpy as np
import pytest
import pytorch_lightning as pl
import torch


@pytest.fixture(autouse=True)
def _isolate_literature_training_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[None]:
    """Pin numerical fits to CPU, isolate artifacts, and restore process globals."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    deterministic = torch.are_deterministic_algorithms_enabled()
    deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    thread_count = torch.get_num_threads()
    trainer_init = pl.Trainer.__init__

    def cpu_trainer_init(self, *args, **kwargs) -> None:
        kwargs["accelerator"] = "cpu"
        kwargs["devices"] = 1
        trainer_init(self, *args, **kwargs)

    monkeypatch.setattr(pl.Trainer, "__init__", cpu_trainer_init)
    monkeypatch.chdir(tmp_path)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.use_deterministic_algorithms(deterministic, warn_only=deterministic_warn_only)
        torch.set_num_threads(thread_count)
