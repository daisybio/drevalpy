"""Private featurizer/predictor execution stack for DRPModel."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from drevalpy.components.feature_block import FeatureBlock
from drevalpy.components.featurizer_fit_context import FeaturizerFitContext
from drevalpy.components.featurizer_label import qualified_featurizer_selector
from drevalpy.components.featurizers._matrix import unique_entity_ids
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.model_input_build import build_model_input_batch
from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.training_context import TrainingContext
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.config import FeaturizerConfig, ModelConfig, PredictionMode
from drevalpy.models.config.resolved import ResolvedModelConfig


def _build_fit_context(
    response: DrugResponseDataset,
    *,
    early_stopping: DrugResponseDataset | None,
    side: Literal["cell_line", "drug"],
) -> FeaturizerFitContext:
    if side == "cell_line":
        train_ids = response.cell_line_ids
        es_ids = early_stopping.cell_line_ids if early_stopping is not None else np.array([], dtype=str)
    elif side == "drug":
        train_ids = response.drug_ids
        es_ids = early_stopping.drug_ids if early_stopping is not None else np.array([], dtype=str)
    else:
        msg = f"Unknown featurizer side {side!r}"
        raise ValueError(msg)
    return FeaturizerFitContext(
        unique_train_ids=unique_entity_ids(train_ids),
        pair_expanded_train_ids=np.asarray(train_ids, dtype=str),
        unique_early_stopping_ids=unique_entity_ids(es_ids) if es_ids.size else np.array([], dtype=str),
        pair_expanded_early_stopping_ids=np.asarray(es_ids, dtype=str) if es_ids.size else np.array([], dtype=str),
        side=side,
    )


def _entity_id_only_featurizer(featurizer: Featurizer | None) -> bool:
    return getattr(featurizer, "entity_id_only", False)


def _empty_feature_dataset() -> FeatureDataset:
    return FeatureDataset(features={})


def _instantiate_featurizer(
    config: FeaturizerConfig,
    resolved: ResolvedModelConfig,
) -> Featurizer:
    registry = str(config.registry)
    if config.name == "concatFeaturizers":
        children = [_instantiate_featurizer(child, resolved) for child in (config.featurizers or ())]
        return config.create_instance({"featurizers": children})
    selector = qualified_featurizer_selector(config.name, config.view)
    return config.create_instance(resolved.featurizer_values(registry, selector))


def build_component_stack(config: ModelConfig | ResolvedModelConfig) -> _ComponentStack:
    """Instantiate featurizers and predictor for a validated config.

    :param config: Template or resolved model configuration.
    :returns: Component stack ready for training.
    """
    from drevalpy.components.tuning.search_space import resolve_model_config

    resolved = config if isinstance(config, ResolvedModelConfig) else resolve_model_config(config)
    template = resolved.template
    cell_line = (
        _instantiate_featurizer(template.cell_line_featurizer, resolved)
        if template.cell_line_featurizer is not None
        else None
    )
    drug = _instantiate_featurizer(template.drug_featurizer, resolved) if template.drug_featurizer is not None else None
    predictor_hp: dict[str, Any] = {
        **resolved.predictor_values(),
        "prediction_mode": template.prediction_mode,
    }
    predictor = template.predictor.create_instance(predictor_hp)
    return _ComponentStack(
        cell_line,
        drug,
        predictor,
        prediction_mode=template.prediction_mode,
        resolved=resolved,
    )


class _ComponentStack:
    """Fit featurizers on training entities, then train a predictor on featurized pairs."""

    def __init__(
        self,
        cell_line_featurizer: Featurizer | None,
        drug_featurizer: Featurizer | None,
        predictor: Predictor,
        *,
        prediction_mode: PredictionMode = PredictionMode.REGRESSION,
        resolved: ResolvedModelConfig,
    ) -> None:
        self._cell_line_featurizer = cell_line_featurizer
        self._drug_featurizer = drug_featurizer
        self._predictor = predictor
        self._prediction_mode = prediction_mode
        self._resolved = ResolvedModelConfig.model_validate(resolved.model_dump(mode="python"))
        self._cell_line_matrix: np.ndarray | None = None
        self._drug_matrix: np.ndarray | None = None
        self._cell_line_entity_ids: np.ndarray | None = None
        self._drug_entity_ids: np.ndarray | None = None

    @property
    def config(self) -> ModelConfig:
        """Return the immutable template associated with this stack.

        :returns: Template ``ModelConfig``.
        """
        return self._resolved.template

    @property
    def resolved(self) -> ResolvedModelConfig:
        """Return the resolved instance configuration.

        :returns: ``ResolvedModelConfig``.
        """
        return self._resolved

    def _build_batch(
        self,
        response: DrugResponseDataset,
        *,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None,
        cell_line_entity_ids: np.ndarray,
        drug_entity_ids: np.ndarray | None,
        cell_line_matrix: np.ndarray,
        drug_matrix: np.ndarray | None,
        output_earlystopping: DrugResponseDataset | None = None,
        training_context: TrainingContext | None = None,
    ) -> ModelInputBatch:
        cell_line_blocks: dict[str, FeatureBlock] = {}
        if self._cell_line_featurizer is not None:
            cell_line_blocks = self._cell_line_featurizer.transform_blocks(
                cell_line_input,
                cell_line_entity_ids,
            )

        drug_blocks: dict[str, FeatureBlock] = {}
        if self._drug_featurizer is not None and drug_entity_ids is not None:
            drug_source = drug_input if drug_input is not None else _empty_feature_dataset()
            if drug_input is None and not _entity_id_only_featurizer(self._drug_featurizer):
                msg = "drug_input is required when a drug featurizer is configured"
                raise ValueError(msg)
            drug_blocks = self._drug_featurizer.transform_blocks(drug_source, drug_entity_ids)

        return build_model_input_batch(
            response,
            cell_line_entity_ids=cell_line_entity_ids,
            drug_entity_ids=drug_entity_ids if self._drug_featurizer is not None else None,
            cell_line_features=cell_line_matrix,
            drug_features=drug_matrix if self._drug_featurizer is not None else None,
            cell_line_blocks=cell_line_blocks,
            drug_blocks=drug_blocks,
            early_stopping_response=output_earlystopping,
            training_context=training_context,
        )

    def _require_drug_input(self, drug_input: FeatureDataset | None) -> FeatureDataset:
        drug_source = drug_input if drug_input is not None else _empty_feature_dataset()
        if drug_input is None and not _entity_id_only_featurizer(self._drug_featurizer):
            msg = "drug_input is required when a drug featurizer is configured"
            raise ValueError(msg)
        return drug_source

    def _fit_transform_featurizer(
        self,
        featurizer: Featurizer,
        source: FeatureDataset,
        *,
        train_entity_ids: np.ndarray,
        feature_input: FeatureDataset | None,
        entity_id_only_ids: np.ndarray,
        fit_context: FeaturizerFitContext | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        featurizer.fit(source, entity_ids=train_entity_ids, context=fit_context)
        if _entity_id_only_featurizer(featurizer):
            entity_ids = np.asarray(entity_id_only_ids, dtype=str)
        elif feature_input is not None:
            entity_ids = np.array(list(feature_input.features.keys()), dtype=str)
        else:
            entity_ids = np.asarray(train_entity_ids, dtype=str)
        matrix = featurizer.transform(source, entity_ids)
        return entity_ids, matrix

    def _train_cell_line_side(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        *,
        output_earlystopping: DrugResponseDataset | None = None,
    ) -> None:
        if self._cell_line_featurizer is None:
            self._cell_line_entity_ids = np.array([], dtype=str)
            self._cell_line_matrix = np.empty((0, 0), dtype=np.float32)
            return
        train_cell_lines = unique_entity_ids(output.cell_line_ids)
        fit_context = _build_fit_context(
            output,
            early_stopping=output_earlystopping,
            side="cell_line",
        )
        entity_ids, matrix = self._fit_transform_featurizer(
            self._cell_line_featurizer,
            cell_line_input,
            train_entity_ids=train_cell_lines,
            feature_input=cell_line_input,
            entity_id_only_ids=train_cell_lines,
            fit_context=fit_context,
        )
        self._cell_line_entity_ids = entity_ids
        self._cell_line_matrix = matrix

    def _train_drug_side(
        self,
        output: DrugResponseDataset,
        drug_input: FeatureDataset | None,
        *,
        output_earlystopping: DrugResponseDataset | None = None,
    ) -> None:
        if self._drug_featurizer is None:
            self._drug_entity_ids = np.array([], dtype=str)
            self._drug_matrix = np.empty((0, 0), dtype=np.float32)
            return
        train_drugs = unique_entity_ids(output.drug_ids)
        drug_source = self._require_drug_input(drug_input)
        fit_context = _build_fit_context(
            output,
            early_stopping=output_earlystopping,
            side="drug",
        )
        entity_ids, matrix = self._fit_transform_featurizer(
            self._drug_featurizer,
            drug_source,
            train_entity_ids=train_drugs,
            feature_input=drug_input,
            entity_id_only_ids=train_drugs,
            fit_context=fit_context,
        )
        self._drug_entity_ids = entity_ids
        self._drug_matrix = matrix

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        *,
        output_earlystopping: DrugResponseDataset | None = None,
        training_context: TrainingContext | None = None,
    ) -> _ComponentStack:
        if len(output) == 0:
            return self

        self._train_cell_line_side(output, cell_line_input, output_earlystopping=output_earlystopping)
        self._train_drug_side(output, drug_input, output_earlystopping=output_earlystopping)

        cell_line_entity_ids = (
            self._cell_line_entity_ids if self._cell_line_entity_ids is not None else np.array([], dtype=str)
        )
        cell_line_matrix = (
            self._cell_line_matrix if self._cell_line_matrix is not None else np.empty((0, 0), dtype=np.float32)
        )
        batch = self._build_batch(
            output,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
            cell_line_entity_ids=cell_line_entity_ids,
            drug_entity_ids=self._drug_entity_ids,
            cell_line_matrix=cell_line_matrix,
            drug_matrix=self._drug_matrix,
            output_earlystopping=output_earlystopping,
            training_context=training_context,
        )
        self._predictor.fit(batch)
        return self

    def is_fitted(self) -> bool:
        """Return whether the predictor has fitted state.

        :returns: ``True`` when the predictor has been fitted.
        """
        return self._predictor.is_fitted()

    def component_state(self) -> dict[str, object]:
        """Return serializable state owned by the component stack.

        :returns: Mapping with predictor and featurizer state dicts.
        """
        return {
            "predictor": self._predictor.get_state(),
            "cell_line_featurizer": (
                self._cell_line_featurizer.get_state() if self._cell_line_featurizer is not None else {}
            ),
            "drug_featurizer": self._drug_featurizer.get_state() if self._drug_featurizer is not None else {},
        }

    def restore_component_state(self, state: dict[str, object]) -> None:
        """Restore state produced by ``component_state``.

        :param state: Serialized component state mapping.
        :raises ValueError: If predictor or featurizer state is not a mapping.
        """
        predictor_state = state.get("predictor", {})
        if not isinstance(predictor_state, dict):
            raise ValueError("predictor state is not a mapping")
        self._predictor.set_state(predictor_state)
        for key, featurizer in (
            ("cell_line_featurizer", self._cell_line_featurizer),
            ("drug_featurizer", self._drug_featurizer),
        ):
            value = state.get(key, {})
            if featurizer is not None:
                if not isinstance(value, dict):
                    raise ValueError(f"{key} state is not a mapping")
                featurizer.set_state(value)

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        if not self.is_fitted():
            msg = "Model has not been trained; call train() or load() before predict()"
            raise RuntimeError(msg)
        if len(cell_line_ids) == 0:
            return np.array([])

        response = DrugResponseDataset(
            response=np.zeros(len(cell_line_ids)),
            cell_line_ids=cell_line_ids,
            drug_ids=drug_ids,
        )
        cell_line_entity_ids = np.array([], dtype=str)
        cell_line_matrix = np.empty((0, 0), dtype=np.float32)
        if self._cell_line_featurizer is not None:
            cell_line_entity_ids = unique_entity_ids(cell_line_ids)
            cell_line_matrix = self._cell_line_featurizer.transform(cell_line_input, cell_line_entity_ids)

        drug_entity_ids: np.ndarray | None = None
        drug_matrix: np.ndarray | None = None
        if self._drug_featurizer is not None:
            drug_entity_ids = unique_entity_ids(drug_ids)
            drug_source = drug_input if drug_input is not None else _empty_feature_dataset()
            if drug_input is None and not _entity_id_only_featurizer(self._drug_featurizer):
                msg = "drug_input is required when a drug featurizer is configured"
                raise ValueError(msg)
            drug_matrix = self._drug_featurizer.transform(drug_source, drug_entity_ids)

        batch = self._build_batch(
            response,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
            cell_line_entity_ids=cell_line_entity_ids,
            drug_entity_ids=drug_entity_ids,
            cell_line_matrix=cell_line_matrix,
            drug_matrix=drug_matrix,
        )
        return self._predictor.predict(batch)
