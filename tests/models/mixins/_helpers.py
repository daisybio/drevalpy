"""Shared checkpoint fixtures for the persistence mixin tests."""

from __future__ import annotations

import io
import zipfile
from typing import Any

import joblib
from upath import UPath

from drevalpy.models import construct_model
from drevalpy.models.config import from_spec
from drevalpy.models.mixins._persistence_io import FORMAT_NAME, FORMAT_VERSION, PAYLOAD_MEMBER
from tests.models.synthetic_fixtures import (
    lco_split_masks,
    synthetic_mudataset_gene_expression_fingerprints,
)


def fitted_elastic_net():
    """Train the cheapest real model in the zoo on the 2x2 synthetic dataset.

    :returns: A fitted ``ElasticNet`` model whose stack reports ``is_fitted()``.
    """
    model = construct_model("ElasticNet")({"alpha": 0.1, "l1_ratio": 0.5})
    model.train(synthetic_mudataset_gene_expression_fingerprints(), lco_split_masks())
    return model


def write_archive(archive_path: UPath, payload: object) -> UPath:
    """Write *payload* into a checkpoint-shaped zip archive.

    :param archive_path: Destination archive path.
    :param payload: Object to serialize as the archive's payload member.
    :returns: *archive_path*.
    """
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    joblib.dump(payload, buffer)
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(PAYLOAD_MEMBER, buffer.getvalue())
    return archive_path


def elastic_net_payload(state: Any, *, model_name: str = "ElasticNet") -> dict[str, Any]:
    """Build a well-formed checkpoint payload carrying an arbitrary *state*.

    :param state: Value stored under the payload's ``state`` key.
    :param model_name: Model identity recorded in the payload.
    :returns: Checkpoint payload mapping.
    """
    config = from_spec("ElasticNet", hyperparameters={"alpha": 0.1, "l1_ratio": 0.5})
    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "model_name": model_name,
        "config": config.model_dump(mode="json"),
        "state": state,
    }
