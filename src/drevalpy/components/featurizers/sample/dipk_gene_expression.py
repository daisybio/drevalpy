"""DIPK gene-expression autoencoder featurizer.

The autoencoder in
``drevalpy.components.predictors.literature.dipk.gene_expression_encoder``
defines ``torch.nn.Module`` subclasses, so importing it imports ``torch``.
``drevalpy.registry`` imports this module to register the
``dipkGeneExpression`` featurizer on ``import drevalpy``, so the three symbols are
pulled in inside the methods that use them and re-exported lazily below so this
module path keeps resolving. See ``tests/test_import_cost_policy.py``.

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

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy._utils.torch_io import load_state_dict, save_state_dict
from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers.sample.base import DenseViewSampleFeaturizer
from drevalpy.components.featurizers.sample.gene_lists import select_genes
from drevalpy.registry.sample_featurizer import register
from drevalpy.types.data.batch.feature_block import BlockSpec
from drevalpy.types.data.feature_source import FeatureSource
from drevalpy.types.data.views import find_view

if TYPE_CHECKING:
    from drevalpy.components.predictors.literature.dipk.gene_expression_encoder import GeneExpressionEncoder

_RE_EXPORTED = frozenset({"GeneExpressionEncoder", "encode_gene_expression", "train_gene_expession_autoencoder"})

#: ``expression_unit`` values that are linear and therefore get ``log2(x + 1)`` first.
_LINEAR_UNITS = frozenset({"tpm", "rpkm", "fpkm", "cpm"})


def __getattr__(name: str) -> Any:
    """Resolve the lazily re-exported autoencoder symbols on first access.

    :param name: Attribute being looked up on this module.
    :returns: The requested attribute, imported on demand.
    :raises AttributeError: If *name* is not one of the re-exported symbols.
    """
    if name in _RE_EXPORTED:
        from drevalpy.components.predictors.literature.dipk import gene_expression_encoder

        return getattr(gene_expression_encoder, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def _is_linear_expression(source: FeatureSource, view: str) -> bool:
    """Tell whether *view* of *source* records a linear (not log-scale) expression unit.

    :param source: Feature source, possibly without a backing MuData.
    :param view: Name of the gene-expression view.
    :returns: ``True`` only when ``expression_unit`` names a linear unit; a missing unit counts as log scale.
    """
    mdata = source.mdata
    if mdata is None:
        return False
    key = find_view(view, mdata.mod)
    if key is None:
        return False
    return str(mdata.mod[key].uns.get("expression_unit", "")).lower() in _LINEAR_UNITS


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


@register(
    "dipkGeneExpression",
    description="DIPK gene-expression autoencoder embeddings.",
    output_format=FeatureFormat.NUMERIC_MATRIX,
)
class DIPKGeneExpressionFeaturizer(DenseViewSampleFeaturizer):
    """Encode intersection genes into the 512-dimensional DIPK representation.

    Each sample profile is z-scored (``ddof=1``) after a ``log2(x + 1)`` step that applies only to
    linear expression units, then encoded by an autoencoder fitted on the unique training samples.
    """

    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),)
    requires_early_stopping: ClassVar[bool] = True
    input_views: ClassVar[tuple[str, ...]] = ("gene_expression",)
    requires_fit: ClassVar[bool] = True

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
        super().__init__()
        self._gene_list = gene_list
        self._epochs = int(epochs_autoencoder)
        self._lr = float(lr_autoencoder)
        self._patience = int(patience_autoencoder)
        self._batch_size = int(batch_size_autoencoder)
        self._encoder: GeneExpressionEncoder | None = None
        self._input_dim = 0
        self._latent_dim = 512
        self._validation_ids: np.ndarray | None = None

    def _names_to_fit(self, source: FeatureSource) -> tuple[str, ...] | None:
        """Bind the encoder to the genes of the gene list, or to the whole view without one.

        :param source: Feature source providing view matrices.
        :returns: Gene names in gene-list order, or every feature name of the view.
        """
        if self._gene_list is None:
            return super()._names_to_fit(source)
        return select_genes(source, self._view, self._gene_list)

    def _fetch_hyperparameters(self) -> dict[str, Any] | None:
        """Match a stored variant only when it was built for the same gene list.

        :returns: The gene-list setting, or ``None`` for the whole view.
        """
        return None if self._gene_list is None else {"gene_list": self._gene_list}

    def _fit_entity_ids(
        self,
        source: FeatureSource,
        entity_ids: np.ndarray | None,
        pair_expanded_ids: np.ndarray | None,
        pair_expanded_es_ids: np.ndarray | None,
    ) -> np.ndarray:
        """Fit on the unique training IDs, remembering the unique early-stopping IDs.

        :param source: Feature source providing view matrices.
        :param entity_ids: Unused; IDs come from *pair_expanded_ids*.
        :param pair_expanded_ids: Training entity IDs with duplicates per response pair.
        :param pair_expanded_es_ids: Early-stopping entity IDs with duplicates.
        :returns: The sorted unique training IDs.
        :raises ValueError: If *pair_expanded_ids* is missing.
        """
        _ = source, entity_ids
        if pair_expanded_ids is None:
            raise ValueError("dipkGeneExpression requires pair_expanded_ids")
        self._validation_ids = None if pair_expanded_es_ids is None else np.unique(pair_expanded_es_ids)
        return np.unique(pair_expanded_ids)

    def _preprocess(self, source: FeatureSource, matrix: np.ndarray) -> np.ndarray:
        """Apply the log2 step (linear units only) and the per-sample z-score.

        :param source: Feature source the matrix came from; its ``expression_unit`` decides the log step.
        :param matrix: Raw gene-expression matrix, already restricted to the fitted genes.
        :returns: Matrix of z-scored profiles.
        """
        if _is_linear_expression(source, self._view):
            matrix = np.log2(matrix + 1)
        return _zscore_rows(matrix)

    def _fit_state(self, source: FeatureSource, entity_ids: np.ndarray) -> int:
        """Train the DIPK autoencoder and return its latent width.

        :param source: Feature source providing view matrices.
        :param entity_ids: Unique training entity IDs.
        :returns: Latent dimensionality of the trained encoder.
        :raises ValueError: If the train or early-stopping ID set is missing or empty.
        """
        from drevalpy.components.predictors.literature.dipk._autoencoder_batches import avoid_singleton_batch
        from drevalpy.components.predictors.literature.dipk.gene_expression_encoder import (
            train_gene_expession_autoencoder,
        )

        validation_ids = self._validation_ids
        if validation_ids is None or len(entity_ids) == 0 or len(validation_ids) == 0:
            raise ValueError("dipkGeneExpression requires non-empty train and early-stopping IDs")
        train = self._preprocess(source, self._raw_matrix(source, entity_ids))
        validation = self._preprocess(source, self._raw_matrix(source, validation_ids))
        train, validation = avoid_singleton_batch(train, validation, self._batch_size)
        self._input_dim = int(train.shape[1])
        self._encoder = train_gene_expession_autoencoder(
            train,
            validation,
            self._epochs,
            lr=self._lr,
            batch_size=self._batch_size,
            patience=self._patience,
        )
        self._latent_dim = int(self._encoder.latent_dim)
        return self._latent_dim

    def _compute_matrix(self, source: FeatureSource, matrix: np.ndarray) -> np.ndarray:
        """Encode *matrix* into DIPK latent vectors.

        Reached only after the ``requires_fit`` gate, so the encoder exists.

        :param source: Feature source the matrix came from.
        :param matrix: Raw gene-expression matrix.
        :returns: Float matrix of latent embeddings.
        """
        from drevalpy.components.predictors.literature.dipk.gene_expression_encoder import encode_gene_expression

        return encode_gene_expression(self._preprocess(source, matrix), self._encoder)

    def _block_feature_names(self, source: FeatureSource) -> None:
        """Latent dimensions have no names to inherit from the source view.

        :param source: Feature source (unused).
        :returns: Always ``None``.
        """
        _ = source
        return None

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

    def get_state(self) -> dict[str, object]:
        """Serialize fitted encoder weights and shape metadata.

        :returns: State mapping, or an empty dict before fitting.
        """
        if self._encoder is None:
            return {}
        return {
            **self._dense_state(),
            "encoder_state": save_state_dict(self._encoder.state_dict()),
            "input_dim": self._input_dim,
            "latent_dim": self._latent_dim,
            "gene_list": self._gene_list,
            "epochs": self._epochs,
            "epochs_autoencoder": self._epochs,
            "lr_autoencoder": self._lr,
            "patience_autoencoder": self._patience,
            "batch_size_autoencoder": self._batch_size,
        }

    def set_state(self, state: dict[str, object]) -> None:
        """Restore a serialized DIPK encoder from ``get_state``.

        :param state: Mapping previously returned by ``get_state``.
        """
        from drevalpy.components.predictors.literature.dipk.gene_expression_encoder import GeneExpressionEncoder

        blob = state.get("encoder_state")
        input_dim = state.get("input_dim")
        latent_dim = state.get("latent_dim", 512)
        if not isinstance(blob, bytes) or not isinstance(input_dim, int) or not isinstance(latent_dim, int):
            return
        self._input_dim = input_dim
        self._latent_dim = latent_dim
        self._output_dim = latent_dim
        gene_list = state.get("gene_list")
        if gene_list is None or isinstance(gene_list, str):
            self._gene_list = gene_list
        epochs = state.get("epochs", state.get("epochs_autoencoder"))
        if isinstance(epochs, int):
            self._epochs = epochs
        lr = state.get("lr_autoencoder")
        if isinstance(lr, float):
            self._lr = lr
        patience = state.get("patience_autoencoder")
        if isinstance(patience, int):
            self._patience = patience
        batch_size = state.get("batch_size_autoencoder")
        if isinstance(batch_size, int):
            self._batch_size = batch_size
        self._encoder = GeneExpressionEncoder(input_dim, latent_dim=latent_dim)
        self._encoder.load_state_dict(load_state_dict(blob))
        self._encoder.eval()
        self._restore_dense_state(state)
        self._is_fitted = True
