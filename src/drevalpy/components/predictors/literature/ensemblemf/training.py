"""Training loop of one EnsembleMF member.

The loop encodes only the samples and drugs that occur in each mini-batch, keeps the
weights with the best validation error and stops after ``patience`` epochs without
improvement.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from drevalpy.components.predictors.literature.ensemblemf.network import MFNet

#: Smallest validation improvement that resets the patience counter.
MIN_IMPROVEMENT = 1e-6
#: Largest gradient norm kept by the clipping step.
MAX_GRAD_NORM = 5.0


@dataclass(frozen=True)
class PairTensors:
    """Training or validation pairs as row indices into the entity feature matrices."""

    sample_idx: torch.Tensor
    drug_idx: torch.Tensor
    response: torch.Tensor

    def __len__(self) -> int:
        """Return the number of pairs."""
        return len(self.response)


@dataclass(frozen=True)
class MemberData:
    """Everything one member trains on."""

    x_sample: torch.Tensor
    x_drug: torch.Tensor
    train: PairTensors
    validation: PairTensors


@dataclass(frozen=True)
class MemberSettings:
    """Optimizer and stopping settings of one member."""

    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int


@torch.no_grad()
def validation_mse(net: MFNet, data: MemberData) -> float:
    """Return the mean squared error of *net* on the validation pairs.

    :param net: Network to evaluate; switched to evaluation mode.
    :param data: Entity features and validation pairs.
    :returns: Mean squared error, or infinity when there are no validation pairs.
    """
    pairs = data.validation
    if len(pairs) == 0:
        return float("inf")
    net.eval()
    z_sample, z_drug = net.encode(data.x_sample, data.x_drug)
    prediction = net.score_pairs(z_sample[pairs.sample_idx], z_drug[pairs.drug_idx])
    return float(nn.functional.mse_loss(prediction, pairs.response).item())


def _train_epoch(net: MFNet, data: MemberData, optimizer: torch.optim.Optimizer, batch_size: int) -> None:
    """Run one shuffled pass over the training pairs.

    :param net: Network to update in place.
    :param data: Entity features and training pairs.
    :param optimizer: Optimizer stepped after every mini-batch.
    :param batch_size: Number of pairs per mini-batch.
    """
    pairs = data.train
    loss_fn = nn.MSELoss()
    net.train()
    perm = torch.randperm(len(pairs), device=pairs.response.device)
    for start in range(0, len(pairs), batch_size):
        idx = perm[start : start + batch_size]
        uniq_sample, inv_sample = torch.unique(pairs.sample_idx[idx], return_inverse=True)
        uniq_drug, inv_drug = torch.unique(pairs.drug_idx[idx], return_inverse=True)
        optimizer.zero_grad()
        drug_id_emb_rows = net.drug_id_emb.weight[uniq_drug] if net.use_drug_id_embedding else None
        z_sample, z_drug = net.encode(data.x_sample[uniq_sample], data.x_drug[uniq_drug], drug_id_emb_rows)
        loss = loss_fn(net.score_pairs(z_sample[inv_sample], z_drug[inv_drug]), pairs.response[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
        optimizer.step()


def train_member(net: MFNet, data: MemberData, settings: MemberSettings) -> None:
    """Train one ensemble member in place and restore its best-validation weights.

    :param net: Network to train.
    :param data: Entity features, training pairs and early-stopping pairs.
    :param settings: Optimizer and stopping settings.
    """
    optimizer = torch.optim.Adam(net.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    best_val, best_state, stale = float("inf"), None, 0
    for _epoch in range(settings.max_epochs):
        _train_epoch(net, data, optimizer, settings.batch_size)
        val_mse = validation_mse(net, data)
        if val_mse < best_val - MIN_IMPROVEMENT:
            best_val, stale = val_mse, 0
            best_state = {key: value.detach().clone() for key, value in net.state_dict().items()}
            continue
        stale += 1
        if stale >= settings.patience:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
