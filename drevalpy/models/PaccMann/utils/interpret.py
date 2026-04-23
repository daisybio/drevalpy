"""Utility functions for uncertainty estimation in PaccMann models."""

import torch
from torch import Tensor, nn

from .utils import get_device

# We use standard deviation to measure uncertainty since entropy is not
# defined for continuous variables and differential entropy is not ideal.
# In case all predictions are identical, std is 0. If 50% are 0 and 50% are
# one, it is maximal, i.e. 0.5.
MAX_STD = 0.5
MIN_STD = 0.0

DEVICE = get_device()


def map_to_device(inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    """Move all input tensors to the configured device.

    :param inputs: Tuple of input tensors
    :return: Tuple of tensors on the target device
    """
    return tuple(x.to(DEVICE) for x in inputs)


def monte_carlo_dropout(model, regime="loader", loader=None, tensors=None, repetitions=20):  # noqa C901
    """Attempts to approximate epistemic uncertainty through MC dropout.

    Performs Monte Carlo dropout for a given model and returns a list of
    sample-wise confidence estimates.
    This method can be used in two regimes, either by passing a dataloader
    or by passing a tensor with the raw input to the model.

    :param model: Torch model to evaluate
    :param regime: Either 'loader' or 'tensors'
    :param loader: The dataset to be tested
        The loader is expected to return a tuple with the last item
        being the labels and all others the model inputs.
        Is only used if 'regime'=='loader'
    :param tensors: The input tensor(s) for the model
        Can either be a single tensor or a tuple of tensors (in the right order)
    :param repetitions: Amount of forward passes for each sample

    :return: Tuple (confidences, predictions) where confidences contain the inverse
        normalized standard deviation of the MC dropout estimates.
    :raises ValueError: If regime is invalid or tensor has an invalid type.
    :raises AttributeError: If the loader does not use sequential sampling.
    """
    if regime != "loader" and regime != "tensors":
        raise ValueError("Choose regime from {'loader', 'tensors'}")

    # Activate dropout layers while keeping other rest in eval mode.
    def enable_dropout(m):
        if isinstance(m, nn.Dropout):
            m.train()

    model.eval()
    model.apply(enable_dropout)

    if regime == "loader":

        # Error handling
        if not isinstance(loader.sampler, torch.utils.data.sampler.SequentialSampler):
            raise AttributeError(
                "Data loader does not use sequential sampling. Consider set"
                "ting shuffle=False when instantiating the data loader."
            )

        # Run over all batches in the loader

        def call_fn():
            preds = []
            for inputs in loader:
                # inputs is a tuple with the last element being the labels
                # outs can be a n-tuple returned by the model
                outs = model(*map_to_device(inputs[:-1]))
                preds.append(outs[0].detach().cpu() if isinstance(outs, tuple) else outs.detach().cpu())

            return torch.cat(preds)

    elif regime == "tensors":

        if not isinstance(tensors, tuple) and not isinstance(tensors, torch.Tensor):
            raise ValueError("Tensor needs to either tuple or torch.Tensor")

        inputs = tensors if isinstance(tensors, tuple) else (tensors,)

        def call_fn():
            outs = model(*map_to_device(inputs))
            return outs[0] if isinstance(outs, tuple) else outs

    with torch.no_grad():
        predictions = [torch.unsqueeze(call_fn(), -1) for _ in range(repetitions)]
    predictions = torch.cat(predictions, dim=-1)

    # Scale confidences to [0, 1]
    confidences = -1 * ((predictions.std(dim=-1) - MIN_STD) / (MAX_STD - MIN_STD)) + 1

    model.eval()

    return confidences, torch.mean(predictions, -1)


def test_time_augmentation(  # noqa: C901
    model,
    regime="loader",
    loader=None,
    tensors=None,
    repetitions=20,
    augmenter=None,
    tensors_to_augment=None,
):
    """Attempts to measure aleatoric uncertainty through augmentation during test time.

    It returns a list of sample-wise confidence estimates.

    This method can be used in two regimes, either by passing a dataloader
    or by passing a tensor with the raw input to the model.

    :param model: The torch network to be investigated.
    :param regime: Either 'loader' or 'tensors'
    :param loader: The dataset to be tested
        The loader is expected to return a tuple with the last item
        being the labels and all others the model inputs. The loader should
        natively perform data augmentation.
        Is only used if 'regime'=='loader'.
    :param tensors: The input tensor(s) for the model
        Can either be a single tensor or a tuple of tensors (in the
        right order)
    :param repetitions: Amount of forward passes for each sample
    :param augmenter: This can either be function that performs the augmentation,
        e.g. an object of type
        pytoda.smiles.AugmentTensor (if `tensors` represents a SMILES
        tensor). Alternatively, it can also be a list of augmenters with
        the same length like tensors_to_augment.
        Only used if regime=='tensors'.
    :param tensors_to_augment: This can either be an integer
        pointing to the tensor to be augmented. E.g. tensors_to_augment = 0
        augments the first tensor in tensors. Can also be a list of the
        same length as augmenter (if several augmentations should be
        performed  on several tensors simultaneously).
        Only used if regime=='tensors'.

    :return: Tuple (confidences, predictions) where confidences contains
        inverse normalized standard deviations and predictions contains mean
        predictions across repetitions.
    :raises ValueError: If regime is invalid, tensor inputs are invalid,
        augmentation indices are invalid or the number of augmenters does
        not match the number of tensors to augment.
    :raises AttributeError: If the loader does not use sequential sampling.
    """
    if regime != "loader" and regime != "tensors":
        raise ValueError("Choose regime from {'loader', 'tensors'}")

    model.eval()

    if regime == "loader":

        # Error handling
        if not isinstance(loader.sampler, torch.utils.data.sampler.SequentialSampler):
            raise AttributeError(
                "Data loader does not use sequential sampling. Consider set"
                "ting shuffle=False when instantiating the data loader."
            )

        # Run over all batches in the loader

        def call_fn():
            preds = []
            for inputs in loader:
                # inputs is a tuple with the last element being the labels
                # outs can be a n-tuple returned by the model
                outs = model(*map_to_device(inputs[:-1]))
                preds.append(outs[0] if isinstance(outs, tuple) else outs)

            return torch.cat(preds)

    elif regime == "tensors":

        if not isinstance(tensors, tuple) and not isinstance(tensors, torch.Tensor):
            raise ValueError("Tensor needs to either tuple or torch.Tensor")
        if not isinstance(tensors_to_augment, list) and not isinstance(tensors_to_augment, int):
            raise ValueError("tensors_to_augment needs to be list or int")

        # Convert input to common formats (tuples and lists)
        tensors_to_augment = [tensors_to_augment] if isinstance(tensors_to_augment, int) else tensors_to_augment
        inputs = tensors if isinstance(tensors, tuple) else (tensors,)
        aug_fns = augmenter if isinstance(augmenter, tuple) else (augmenter,)

        # Error handling
        if not len(aug_fns) == len(tensors_to_augment):
            raise ValueError("Provide one augmenter for each tensor you want to augment.")
        if max(tensors_to_augment) > len(inputs):
            raise ValueError(
                "tensors_to_augment should be indexes to the tensors used for "
                f"augmentation. {max(tensors_to_augment)} is larger than "
                f"length of inputs ({len(inputs)})."
            )

        def call_fn():
            # Perform augmentation on all designated functions
            augmented_inputs = [
                (aug_fns[tensors_to_augment.index(ind)](tensor) if ind in tensors_to_augment else tensor)
                for ind, tensor in enumerate(inputs)
            ]
            outs = model(*map_to_device(augmented_inputs))
            return outs[0] if isinstance(outs, tuple) else outs

    with torch.no_grad():
        predictions = [torch.unsqueeze(call_fn(), -1) for _ in range(repetitions)]
    predictions = torch.cat(predictions, dim=-1)

    # Scale confidences to [0, 1]
    confidences = -1 * ((predictions.std(dim=-1) - MIN_STD) / (MAX_STD - MIN_STD)) + 1

    return torch.clamp(confidences, min=0), torch.mean(predictions, -1)
