"""DIPK gene-expression autoencoder featurizer.

The autoencoder in
``drevalpy.components.predictors.literature.dipk.gene_expression_encoder``
defines ``torch.nn.Module`` subclasses, so importing it imports ``torch``.
``drevalpy.registry`` imports this module to register the
``dipkGeneExpression`` featurizer on ``import drevalpy``, so the autoencoder symbols are
pulled in inside the methods that use them. See ``tests/test_import_cost_policy.py``.

The featurizer follows drevalpy 1.6.1: the autoencoder is fitted on the *unique* samples
(and unique early-stopping samples), and every sample profile is z-scored on its own
(``ddof=1``) after the gene selection, as the published DIPK preparation does.

**Log scale.** 1.6.1 applied ``log2(x + 1)`` to the linear-TPM datasets only. Here the
decision is read from the view's ``expression_unit`` (``uns`` of the ``gene_expression``
modality, written by the curation pipeline): a linear unit (``tpm``, ``rpkm``, ``fpkm``,
``cpm``) gets ``log2(x + 1)`` before the z-score, a log unit such as ``log2_tpm_plus_1``
does not, and a source that records no unit (a test mock, a user dataset) is taken as
already on a log scale, which is what every curated v2 dataset provides. The decision is
made per source on every call, so a model trained on one dataset and applied to another
treats each by its own unit, as 1.6.1 did per dataset name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np

from drevalpy.components.featurizers._dense import dense_result, view_matrix
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.components.featurizers.sample.gene_lists import view_genes
from drevalpy.errors import InvalidInputError
from drevalpy.registry.featurizer import register
from drevalpy.types.data.batch.feature_block import FeaturizerResult
from drevalpy.types.data.feature_source import FeatureSource, Side

if TYPE_CHECKING:
    from drevalpy.components.predictors.literature.dipk.gene_expression_encoder import GeneExpressionEncoder

_VIEW = "gene_expression"

#: ``expression_unit`` values that are linear and therefore get ``log2(x + 1)`` first.
_LINEAR_UNITS = frozenset({"tpm", "rpkm", "fpkm", "cpm"})


def _is_linear_expression(source: FeatureSource, view: str) -> bool:
    """Tell whether *view* of *source* records a linear (not log-scale) expression unit.

    :param source: Feature source.
    :param view: Name of the gene-expression view.
    :returns: ``True`` only when ``expression_unit`` names a linear unit; a missing unit counts as log scale.
    """
    return str(source.expression_unit(view) or "").lower() in _LINEAR_UNITS


def _zscore_rows(matrix: np.ndarray) -> np.ndarray:
    """Z-score every row on its own with the sample standard deviation, like 1.6.1.

    Rows are processed one at a time with 1-D reductions so the summation order, and with it the
    result, equals ``(x - np.mean(x)) / np.std(x, ddof=1)`` of the published preparation.

    :param matrix: Matrix with one sample per row.
    :returns: Matrix of the same shape; integer input is promoted to float64.
    """
    matrix = np.asarray(matrix)
    if not np.issubdtype(matrix.dtype, np.floating):
        matrix = matrix.astype(np.float64)
    out = np.empty_like(matrix)
    for i, row in enumerate(matrix):
        out[i] = (row - np.mean(row)) / np.std(row, ddof=1)
    return out


def dipk_preprocess(source: FeatureSource, matrix: np.ndarray, view: str = "gene_expression") -> np.ndarray:
    """Apply DIPK's expression preparation: ``log2(x + 1)`` for linear units, then a per-sample z-score.

    :param source: Feature source the matrix came from; the ``expression_unit`` of its *view* decides the
        log step.
    :param matrix: Raw gene-expression matrix, one sample per row, already restricted to the encoder's genes.
    :param view: Name of the gene-expression view.
    :returns: Matrix of z-scored profiles.
    """
    if _is_linear_expression(source, view):
        matrix = np.log2(matrix + 1)
    return _zscore_rows(matrix)


@register("dipkGeneExpression", description="DIPK gene-expression autoencoder embeddings.")
class DIPKGeneExpressionFeaturizer(Featurizer):
    """Encode intersection genes into the 512-dimensional DIPK representation.

    Each sample profile is z-scored (``ddof=1``) after a ``log2(x + 1)`` step that applies only to
    linear expression units, then encoded by an autoencoder fitted on the unique training samples.
    """

    sides: ClassVar[frozenset[Side]] = frozenset({"sample"})
    requires_early_stopping: ClassVar[bool] = True
    input_views: ClassVar[tuple[str, ...]] = (_VIEW,)

    def __init__(
        self,
        *,
        gene_list: str | None = None,
        epochs_autoencoder: int = 1000,
        lr_autoencoder: float = 1e-4,
        patience_autoencoder: int = 3,
        batch_size_autoencoder: int = 1024,
    ) -> None:
        """Store the autoencoder training settings.

        The ``_autoencoder`` suffix keeps these apart from the ``lr``, ``patience`` and ``batch_size``
        of the DIPK predictor, which hold different values.

        :param gene_list: Name of a shipped gene list to keep, in list order; the view must hold every gene of
            it. ``None`` keeps the whole view.
        :param epochs_autoencoder: Maximum number of autoencoder training epochs.
        :param lr_autoencoder: Learning rate of the autoencoder's Adam optimizer.
        :param patience_autoencoder: Epochs without improvement before the autoencoder stops.
        :param batch_size_autoencoder: Mini-batch size of the autoencoder.
        """
        self._gene_list = gene_list
        self._epochs = int(epochs_autoencoder)
        self._lr = float(lr_autoencoder)
        self._patience = int(patience_autoencoder)
        self._batch_size = int(batch_size_autoencoder)
        self._names: tuple[str, ...] | None = None
        self._encoder: GeneExpressionEncoder | None = None

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the tunable autoencoder settings.

        :returns: Hyperparameter search space mapping.
        """
        return {
            "epochs_autoencoder": {"type": "int", "low": 1, "high": 2000, "default": 1000},
            "lr_autoencoder": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True, "default": 1e-4},
            "patience_autoencoder": {"type": "int", "low": 1, "high": 20, "default": 3},
            "batch_size_autoencoder": {
                "type": "categorical",
                "choices": [32, 64, 128, 256, 512, 1024],
                "default": 1024,
            },
        }

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Train the DIPK autoencoder on the sorted unique training samples, validated on the early-stopping ones.

        :param source: Feature source of the sample side.
        :param train_ids: Training sample IDs.
        :param train_pair_ids: Unused.
        :param es_pair_ids: Sample ID of every early-stopping pair; their unique samples validate the
            autoencoder.
        :returns: This featurizer, fitted.
        :raises InvalidInputError: If no training or early-stopping sample is given.
        """
        from drevalpy.components.predictors.literature.dipk._autoencoder_batches import avoid_singleton_batch
        from drevalpy.components.predictors.literature.dipk.gene_expression_encoder import (
            train_gene_expession_autoencoder,
        )

        _ = train_pair_ids
        if es_pair_ids is None or len(es_pair_ids) == 0 or len(train_ids) == 0:
            msg = f"{type(self).__name__} requires a usable early-stopping holdout"
            raise InvalidInputError(msg)
        self._names = view_genes(source, _VIEW, self._gene_list)
        train = dipk_preprocess(source, view_matrix(source, _VIEW, np.unique(train_ids), self._names))
        validation = dipk_preprocess(source, view_matrix(source, _VIEW, np.unique(es_pair_ids), self._names))
        train, validation = avoid_singleton_batch(train, validation, self._batch_size)
        self._encoder = train_gene_expession_autoencoder(
            train,
            validation,
            self._epochs,
            lr=self._lr,
            batch_size=self._batch_size,
            patience=self._patience,
        )
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        """Encode the rows of *ids* into DIPK latent vectors.

        :param source: Feature source of the sample side.
        :param ids: Sample IDs, in output order.
        :returns: One unnamed ``gene_expression`` block of latent vectors.
        :raises RuntimeError: If called before ``fit``.
        """
        from drevalpy.components.predictors.literature.dipk.gene_expression_encoder import encode_gene_expression

        if self._encoder is None:
            msg = "DIPKGeneExpressionFeaturizer must be fit before transform"
            raise RuntimeError(msg)
        if len(ids) == 0:
            matrix = np.empty((0, int(self._encoder.latent_dim)))
        else:
            raw = view_matrix(source, _VIEW, ids, self._names)
            matrix = encode_gene_expression(dipk_preprocess(source, raw), self._encoder)
        return dense_result(ids, _VIEW, matrix, None)
