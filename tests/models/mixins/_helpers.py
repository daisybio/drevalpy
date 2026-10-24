"""Shared checkpoint fixtures for the persistence mixin tests."""

from __future__ import annotations

import io
import zipfile
from typing import Any

import joblib
from upath import UPath

from drevalpy.models import DRPModel
from drevalpy.models.config import from_spec
from drevalpy.models.mixins._persistence_io import FORMAT_NAME, FORMAT_VERSION, PAYLOAD_MEMBER
from tests.models.synthetic_fixtures import (
    lso_split_masks,
    small_data_spec,
    synthetic_mudataset_gene_expression_fingerprints,
)


def fitted_elastic_net():
    model = DRPModel(small_data_spec("ElasticNet"), {"alpha": 0.1, "l1_ratio": 0.5})
    model.train(synthetic_mudataset_gene_expression_fingerprints(), lso_split_masks().train)
    return model


def write_archive(archive_path: UPath, payload: object) -> UPath:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    joblib.dump(payload, buffer)
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(PAYLOAD_MEMBER, buffer.getvalue())
    return archive_path


def elastic_net_payload(state: Any, *, model_name: str = "ElasticNet") -> dict[str, Any]:
    config = from_spec("ElasticNet", hyperparameters={"alpha": 0.1, "l1_ratio": 0.5})
    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "model_name": model_name,
        "config": config.model_dump(mode="json"),
        "state": state,
    }
