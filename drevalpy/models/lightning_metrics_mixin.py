"""Mixin class for PyTorch Lightning modules to add R^2 and PCC metrics logging."""

import torch

from ..evaluation import AVAILABLE_METRICS


class RegressionMetricsMixin:
    """
    Mixin class for PyTorch Lightning modules to automatically compute and log R^2 and PCC metrics.

    This mixin provides:
    - Storage for predictions and targets during training/validation steps
    - Automatic computation of R^2 and PCC at epoch end
    - Consistent logging to wandb via PyTorch Lightning's logging system

    Usage:
        class MyModel(RegressionMetricsMixin, pl.LightningModule):
            def __init__(self, ...):
                super().__init__()
                # Initialize your model...
                self._init_metrics_storage()  # Call this in __init__

            def training_step(self, batch, batch_idx):
                # ... your training logic ...
                predictions = self.forward(...)
                loss = self.criterion(predictions, targets)
                self.log("train_loss", loss, ...)
                self._store_predictions(predictions, targets, is_training=True)
                return loss

            def validation_step(self, batch, batch_idx):
                # ... your validation logic ...
                predictions = self.forward(...)
                loss = self.criterion(predictions, targets)
                self.log("val_loss", loss, ...)
                self._store_predictions(predictions, targets, is_training=False)
                return loss
    """

    def _init_metrics_storage(self) -> None:
        """Initialize storage for predictions and targets."""
        self.train_predictions: list[torch.Tensor] = []
        self.train_targets: list[torch.Tensor] = []
        self.val_predictions: list[torch.Tensor] = []
        self.val_targets: list[torch.Tensor] = []

    def _store_predictions(self, predictions: torch.Tensor, targets: torch.Tensor, is_training: bool = True) -> None:
        """
        Store predictions and targets for epoch-end metric computation.

        :param predictions: model predictions tensor
        :param targets: ground truth targets tensor
        :param is_training: whether this is from training (True) or validation (False)
        """
        # Ensure tensors are detached and on CPU for numpy conversion
        preds_cpu = predictions.detach().cpu()
        targets_cpu = targets.detach().cpu()

        if is_training:
            self.train_predictions.append(preds_cpu)
            self.train_targets.append(targets_cpu)
        else:
            self.val_predictions.append(preds_cpu)
            self.val_targets.append(targets_cpu)

    def _compute_epoch_metrics(self, predictions: list[torch.Tensor], targets: list[torch.Tensor]) -> dict[str, float]:
        """
        Compute R^2 and PCC metrics from stored predictions and targets.

        :param predictions: list of prediction tensors from the epoch
        :param targets: list of target tensors from the epoch
        :returns: dictionary with "R^2" and "Pearson" keys, or empty dict if computation fails
        """
        if len(predictions) == 0:
            return {}

        try:
            # Concatenate all predictions and targets from the epoch
            all_preds = torch.cat(predictions).numpy()
            all_targets = torch.cat(targets).numpy()

            # Compute metrics
            r2 = AVAILABLE_METRICS["R^2"](y_pred=all_preds, y_true=all_targets)
            pcc = AVAILABLE_METRICS["Pearson"](y_pred=all_preds, y_true=all_targets)

            return {"R^2": r2, "Pearson": pcc}
        except Exception:
            # If computation fails (e.g., NaN values, insufficient data), return empty dict
            return {}

    def on_train_epoch_end(self) -> None:
        """
        Epoch-end hook for training.

        Intentionally does NOT log R^2/Pearson per epoch anymore. We only keep
        these buffers to allow optional debugging or future extensions.
        """
        # Clear stored predictions/targets for next epoch
        self.train_predictions.clear()
        self.train_targets.clear()

    def on_validation_epoch_end(self) -> None:
        """
        Epoch-end hook for validation.

        Intentionally does NOT log R^2/Pearson per epoch anymore. Final metrics
        are logged once at the end via DRPModel.compute_and_log_final_metrics().
        """
        # Clear stored predictions/targets for next epoch
        self.val_predictions.clear()
        self.val_targets.clear()
