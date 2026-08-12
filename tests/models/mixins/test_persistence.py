"""Tests for the ``DRPPersistenceMixin`` save/load surface on ``DRPModel``.

The archive format itself is covered in ``test_persistence_io.py``; what is
asserted here is the mixin's own behaviour: delegation from ``save``, identity
checking in ``load``, and the state-restoration guards.
"""

from __future__ import annotations

import pytest
from upath import UPath

from drevalpy.models import construct_model
from drevalpy.models.mixins._persistence_io import (
    CorruptedCheckpointError,
    IncompatibleModelCheckpointError,
    save_model,
)
from tests.models.mixins._helpers import elastic_net_payload, fitted_elastic_net, write_archive


class TestSave:
    """``DRPModel.save`` delegates to ``save_model``."""

    def test_writes_an_archive(self, tmp_path) -> None:
        model = fitted_elastic_net()

        model.save(str(UPath(tmp_path) / "elastic_net"))

        assert (UPath(tmp_path) / "elastic_net.zip").is_file()

    def test_rejects_a_directory_path(self, tmp_path) -> None:
        model = fitted_elastic_net()

        with pytest.raises(ValueError, match="not a directory"):
            model.save(str(tmp_path))


class TestLoad:
    """``DRPModel.load`` restores fitted state onto a fresh instance."""

    def test_round_trips_a_fitted_model(self, tmp_path) -> None:
        model = fitted_elastic_net()
        checkpoint = str(UPath(tmp_path) / "elastic_net")
        model.save(checkpoint)

        loaded = construct_model("ElasticNet").load(checkpoint)

        assert loaded._stack is not None
        assert loaded._stack.is_fitted()

    def test_restores_the_resolved_config(self, tmp_path) -> None:
        model = fitted_elastic_net()
        checkpoint = str(UPath(tmp_path) / "elastic_net")
        model.save(checkpoint)

        loaded = construct_model("ElasticNet").load(checkpoint)

        assert loaded._resolved_model_config is not None
        assert loaded._resolved_model_config.predictor_values()["alpha"] == 0.1

    def test_clears_the_empty_training_flag(self, tmp_path) -> None:
        model = fitted_elastic_net()
        checkpoint = str(UPath(tmp_path) / "elastic_net")
        model.save(checkpoint)

        loaded = construct_model("ElasticNet").load(checkpoint)

        assert loaded._empty_training is False

    def test_rejects_a_checkpoint_from_another_model(self, tmp_path) -> None:
        model = fitted_elastic_net()
        checkpoint = str(UPath(tmp_path) / "elastic_net")
        save_model(model, checkpoint)

        with pytest.raises(IncompatibleModelCheckpointError, match="does not match"):
            construct_model("Ridge").load(checkpoint)

    def test_rejects_an_unfitted_checkpoint_state(self, tmp_path) -> None:
        payload = elastic_net_payload({"predictor": {}, "cell_line_featurizer": {}, "drug_featurizer": {}})
        archive = write_archive(UPath(tmp_path) / "elastic_net.zip", payload)

        with pytest.raises(
            CorruptedCheckpointError, match="missing a fitted estimator|did not restore a fitted predictor"
        ):
            construct_model("ElasticNet").load(str(archive))

    def test_rejects_a_non_mapping_predictor_state(self, tmp_path) -> None:
        payload = elastic_net_payload({"predictor": "bad", "cell_line_featurizer": {}, "drug_featurizer": {}})
        archive = write_archive(UPath(tmp_path) / "elastic_net.zip", payload)

        with pytest.raises(CorruptedCheckpointError, match="component state is invalid"):
            construct_model("ElasticNet").load(str(archive))
