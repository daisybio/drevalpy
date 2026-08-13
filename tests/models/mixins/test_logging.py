"""Tests for DRP wandb logging mixin."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from drevalpy.models.mixins._logging import _DRPLoggingMixin

_EXPECTED_DEFINE_METRICS: list[tuple[str, str]] = [
    ("epoch", "max"),
    ("train_loss", "min"),
    ("val_loss", "min"),
    ("train_R^2", "max"),
    ("val_R^2", "max"),
    ("train_Pearson", "max"),
    ("val_Pearson", "max"),
]


class _Stub(_DRPLoggingMixin):
    def __init__(self) -> None:
        self.wandb_project = None
        self.wandb_run = None
        self.wandb_config = None
        self._in_hyperparameter_tuning = False
        self._hp: dict = {}

    @classmethod
    def get_model_name(cls) -> str:
        return "Stub"

    @property
    def hyperparameters(self) -> dict:
        return self._hp


def test_init_wandb_define_metrics_on_success() -> None:
    stub = _Stub()
    mock_run = MagicMock()
    mock_wandb = MagicMock()

    with patch("drevalpy.models.mixins._logging._wandb", return_value=mock_wandb):
        mock_wandb.run = mock_run
        stub.init_wandb("proj")

        mock_wandb.finish.assert_called_once()
        mock_wandb.init.assert_called_once_with(
            project="proj",
            config={},
            name="Stub",
            tags=None,
        )
        assert stub.wandb_run is mock_run
        assert mock_wandb.define_metric.call_count == len(_EXPECTED_DEFINE_METRICS)
        for metric, summary in _EXPECTED_DEFINE_METRICS:
            mock_wandb.define_metric.assert_any_call(metric, summary=summary)


def test_init_wandb_swallows_define_metric_exceptions() -> None:
    stub = _Stub()
    mock_run = MagicMock()
    mock_wandb = MagicMock()

    with patch("drevalpy.models.mixins._logging._wandb", return_value=mock_wandb):
        mock_wandb.run = mock_run
        mock_wandb.define_metric.side_effect = RuntimeError("define_metric failed")
        stub.init_wandb("proj")
