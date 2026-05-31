"""PyTorch device resolution for CurveCurator fitting."""

from __future__ import annotations

import os


def resolve_device(device: str) -> str:
    """Resolve ``auto`` to cuda / mps / cpu; set CUDA alloc conf before first GPU use.

    :param device: Requested PyTorch device string.
    :returns: Concrete device string for CurveCurator fitting.
    """
    # Harmless for CPU/MPS runs; CUDA reads this at first GPU use. expandable_segments lets
    # PyTorch release fragmented VRAM between batched CurveCurator chunks more reliably.
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" not in alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (alloc_conf + ",expandable_segments:True").lstrip(",")

    if device != "auto":
        return device

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def effective_device(requested: str, n_curves: int, gpu_min_curves: int) -> str:
    """Use CPU for small batches where GPU overhead dominates.

    :param requested: Requested PyTorch device string.
    :param n_curves: Number of curves in the chunk.
    :param gpu_min_curves: Minimum curves before using an accelerator.
    :returns: Concrete device string for the chunk.
    """
    if n_curves < gpu_min_curves:
        return "cpu"
    return resolve_device(requested)
