"""Train / predict over a materialized component stack.

The two methods here are what makes a ``DRPModel`` a model rather than a
configuration object, and they are the only ones that read and write
``_stack`` / ``_empty_training``. Keeping them apart from the config and identity
surface on ``DRPModel`` is the same separation ``_persistence.py`` already draws
for checkpoint I/O.

Call-shape resolution lives one module over, in ``_train_args.py``, so ``train``
below is a dispatch between two input forms rather than a run of ``isinstance``
checks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from upath import UPath as Path

from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.models.component_stack import _ComponentStack
from drevalpy.models.mixins._train_args import TrainCallArgs, resolve_train_args
from drevalpy.types import SplitMask, SplitMasks

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin

    from drevalpy.types.data.dataset import Dataset


class DRPTrainingMixin:
    """Fit and score the component stack a ``DRPModel`` was materialized with."""

    _stack: _ComponentStack | None
    _empty_training: bool

    @classmethod
    def get_model_name(cls) -> str:
        """Return the model identity; implemented by ``DRPModel``.

        :raises NotImplementedError: If the subclass does not implement this hook.
        """
        raise NotImplementedError

    def _require_stack(self) -> _ComponentStack:
        """Return the component stack, refusing an unmaterialized model.

        :returns: The materialized component stack.
        :raises RuntimeError: If the model was constructed without a stack.
        """
        if self._stack is None:
            raise RuntimeError("Model has not been constructed with a component stack")
        return self._stack

    def _training_context(self, model_checkpoint_dir: str | Path) -> TrainingContext:
        """Build the context predictors read for checkpoints and log labels.

        :param model_checkpoint_dir: Directory predictors may write checkpoints to.
        :returns: Training context for this model.
        """
        return TrainingContext(
            checkpoint_dir=Path(model_checkpoint_dir),
            logging_metadata={"model_name": self.get_model_name()},
        )

    def train(
        self,
        mudataset_or_output=None,
        split_or_cell_line_input=None,
        drug_input=None,
        *,
        mudataset: Dataset | None = None,
        split: SplitMasks | None = None,
        scope: SplitMask | None = None,
        early_stopping_scope: SplitMask | None = None,
        output=None,
        cell_line_input=None,
        output_earlystopping=None,
        model_checkpoint_dir: str | Path = "checkpoints",
        response_transformation: TransformerMixin | None = None,
    ) -> None:
        """Train the component stack.

        Supports the Dataset path (positional: mudataset, scope/split) and the
        internal path (output, cell_line_input, drug_input).

        :param mudataset_or_output: First positional slot; a ``Dataset`` selects the
            Dataset path, anything else is taken as ``output``.
        :param split_or_cell_line_input: Second positional slot; a ``SplitMask`` or
            ``SplitMasks`` selects the Dataset path, anything else is ``cell_line_input``.
        :param mudataset: Dataset containing response data and all features.
        :param scope: SplitMask defining train indices for this fold.
        :param split: (compat) SplitMasks; converted to SplitMask internally.
        :param early_stopping_scope: Optional SplitMask for early stopping.
        :param output: ResponseBatch for training pairs.
        :param cell_line_input: FeatureSource for cell lines.
        :param drug_input: FeatureSource for drugs, or None.
        :param output_earlystopping: Optional early-stopping dataset.
        :param model_checkpoint_dir: Directory for predictor checkpoints.
        :param response_transformation: Optional fitted transformer applied to the
            training targets. Predictions therefore live in the transformed space and the
            caller is responsible for inverse-transforming them.
        :raises RuntimeError: If the model lacks a component stack.
        :raises TypeError: If neither accepted set of inputs was passed.
        """
        stack = self._require_stack()
        args = resolve_train_args(
            mudataset_or_output,
            split_or_cell_line_input,
            drug_input,
            mudataset=mudataset,
            split=split,
            scope=scope,
            early_stopping_scope=early_stopping_scope,
            output=output,
            cell_line_input=cell_line_input,
        )
        context = self._training_context(model_checkpoint_dir)

        if args.is_dataset_form:
            self._train_on_dataset(stack, args, context, response_transformation)
            return
        if args.is_feature_source_form:
            self._train_on_feature_sources(stack, args, context, output_earlystopping)
            return
        raise TypeError("train() requires either (mudataset, scope) or (output, cell_line_input)")

    def _train_on_dataset(
        self,
        stack: _ComponentStack,
        args: TrainCallArgs,
        context: TrainingContext,
        response_transformation: TransformerMixin | None,
    ) -> None:
        """Fit the stack from a ``Dataset`` and a training scope.

        A scope selecting no non-NaN response leaves the model in the empty-training
        state, where ``predict`` answers NaN instead of raising.

        :param stack: Materialized component stack.
        :param args: Resolved training arguments in the Dataset form.
        :param context: Training context for checkpoints and log labels.
        :param response_transformation: Optional fitted transformer for the targets.
        """
        train_response = _ComponentStack._extract_response_pairs(args.mudataset, args.scope, response_transformation)
        if len(train_response) == 0:
            self._empty_training = True
            return
        self._empty_training = False
        if args.early_stopping_scope is not None:
            stack.train_with_early_stopping(
                args.mudataset,
                args.scope,
                args.early_stopping_scope,
                training_context=context,
                response_transformation=response_transformation,
            )
            return
        stack.train(
            args.mudataset,
            args.scope,
            training_context=context,
            response_transformation=response_transformation,
        )

    def _train_on_feature_sources(
        self,
        stack: _ComponentStack,
        args: TrainCallArgs,
        context: TrainingContext,
        output_earlystopping: Any,
    ) -> None:
        """Fit the stack from a response batch and raw feature sources.

        :param stack: Materialized component stack.
        :param args: Resolved training arguments in the ResponseBatch form.
        :param context: Training context for checkpoints and log labels.
        :param output_earlystopping: Optional early-stopping response batch.
        """
        self._empty_training = len(args.output) == 0
        if self._empty_training:
            return
        stack._fit_featurizers_and_predictor(
            args.output,
            args.cell_line_input,
            args.drug_input,
            output_earlystopping=output_earlystopping,
            training_context=context,
        )

    def predict(
        self,
        mudataset: Dataset | None = None,
        scope_or_split=None,
        *,
        scope: SplitMask | None = None,
        split: SplitMasks | None = None,
    ) -> np.ndarray:
        """Predict responses for the test pairs in a split.

        :param mudataset: Dataset containing all features.
        :param scope_or_split: Positional SplitMask or SplitMasks (compat).
        :param scope: SplitMask with indices to predict on.
        :param split: SplitMasks; test indices used as scope.
        :returns: Predicted response values.
        :raises RuntimeError: If the model is untrained or lacks a component stack.
        :raises TypeError: If required arguments are missing.
        """
        stack = self._require_stack()
        scope = _resolve_predict_scope(scope_or_split, scope=scope, split=split)
        if mudataset is None or scope is None:
            raise TypeError("predict() requires (mudataset, scope) or (mudataset, split)")

        if self._empty_training:
            test_response = _ComponentStack._extract_response_pairs(mudataset, scope)
            return np.full(len(test_response), np.nan)
        if not stack.is_fitted():
            raise RuntimeError("Model has not been trained; call train() or load() before predict()")
        return stack.predict(mudataset, scope)


def _resolve_predict_scope(
    scope_or_split: Any,
    *,
    scope: SplitMask | None,
    split: SplitMasks | None,
) -> SplitMask | None:
    """Pick the mask ``predict`` scores on out of its three accepted spellings.

    :param scope_or_split: Positional ``SplitMask`` or ``SplitMasks``.
    :param scope: Explicit scope keyword, which wins over both others.
    :param split: Full split masks, whose ``test`` mask is used.
    :returns: The mask to predict on, or ``None`` when none was given.
    """
    if scope is not None:
        return scope
    if isinstance(scope_or_split, SplitMask):
        return scope_or_split
    if isinstance(scope_or_split, SplitMasks):
        return scope_or_split.test
    if split is not None:
        return split.test
    return None
