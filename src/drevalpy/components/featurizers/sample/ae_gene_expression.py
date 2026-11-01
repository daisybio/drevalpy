"""Gene-expression autoencoder featurizer of RandomForest-AE-MolGNet.

``aeGeneExpression`` restricts the expression view to a gene list, trains DIPK's gene-expression
autoencoder on the *unique* training samples (the unique early-stopping samples validate it) and
emits the 512-dimensional latent vectors. It follows ``RandomForestAEMolGNet`` of drevalpy 1.6.1,
which differs from ``dipkGeneExpression`` in three ways: there is no per-sample z-score, the
default gene list is ``tcga_autoencoder_genes_available``, and the autoencoder trains longer
(2000 epochs, patience 50). A linear expression unit gets ``log2(x + 1)`` first, as for DIPK.

The trained encoder can be cached on disk (``autoencoder_cache_dir`` or the environment variable
``DREVAL_AE_CACHE``), see :mod:`drevalpy.components.featurizers.sample._ae_cache`.

The autoencoder module defines ``torch.nn.Module`` subclasses, so it is imported inside the
methods that use it; see ``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np

from drevalpy._utils.torch_io import load_state_dict, save_state_dict
from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers._dense import dense_result, view_matrix
from drevalpy.components.featurizers.featurizer import Featurizer, all_nan_rows_mask
from drevalpy.components.featurizers.sample._ae_cache import cache_file, resolve_cache_dir, write_atomically
from drevalpy.components.featurizers.sample.dipk_gene_expression import _is_linear_expression
from drevalpy.components.featurizers.sample.gene_lists import view_genes
from drevalpy.errors import InvalidInputError
from drevalpy.log import get_logger
from drevalpy.registry.featurizer import register
from drevalpy.types.data.batch.feature_block import BlockSpec, FeaturizerResult
from drevalpy.types.data.feature_source import FeatureSource, Side

if TYPE_CHECKING:
    from drevalpy.components.predictors.literature.dipk.gene_expression_encoder import GeneExpressionEncoder

_logger = get_logger(__name__)

#: Default gene list: the genes of the TCGA autoencoder that the curated screens have.
DEFAULT_GENE_LIST = "tcga_autoencoder_genes_available"

_BLOCK = "gene_expression"


def train_or_load_autoencoder(
    train: np.ndarray,
    validation: np.ndarray,
    train_ids: np.ndarray,
    validation_ids: np.ndarray,
    *,
    epochs: int,
    lr: float,
    patience: int,
    batch_size: int,
    gene_list: str | None = None,
    genes: tuple[str, ...] | None = None,
    cache_dir: str | None = None,
) -> GeneExpressionEncoder:
    """Return the gene-expression autoencoder for one fold, read from the cache when possible.

    :param train: Preprocessed training matrix, one row per unique sample.
    :param validation: Preprocessed early-stopping matrix.
    :param train_ids: Sample IDs of the training rows.
    :param validation_ids: Sample IDs of the early-stopping rows.
    :param epochs: Maximum number of training epochs.
    :param lr: Learning rate of the Adam optimizer.
    :param patience: Epochs without improvement before training stops.
    :param batch_size: Mini-batch size.
    :param gene_list: Gene-list name the matrices were restricted to; part of the cache key.
    :param genes: Column names of the matrices; ``None`` keys the cache by column positions.
    :param cache_dir: Cache directory; ``None`` falls back to ``DREVAL_AE_CACHE``, and without that
        nothing is cached.
    :returns: Trained encoder.
    """
    from drevalpy.components.predictors.literature.dipk._autoencoder_batches import avoid_singleton_batch
    from drevalpy.components.predictors.literature.dipk.gene_expression_encoder import (
        GeneExpressionEncoder,
        train_gene_expession_autoencoder,
    )

    directory = resolve_cache_dir(cache_dir)
    path = None
    if directory is not None:
        config = {"gene_list": gene_list, "epochs": epochs, "batch_size": batch_size, "patience": patience, "lr": lr}
        names = genes or tuple(str(i) for i in range(train.shape[1]))
        path = cache_file(directory, config, names, train_ids, validation_ids, (train, validation))
        if path.is_file():
            encoder = GeneExpressionEncoder(int(train.shape[1]))
            encoder.load_state_dict(load_state_dict(path.read_bytes()))
            encoder.eval()
            _logger.info("Gene autoencoder taken from cache: %s", path)
            return encoder
    train, validation = avoid_singleton_batch(train, validation, batch_size)
    encoder = train_gene_expession_autoencoder(
        train, validation, epochs, lr=lr, batch_size=batch_size, patience=patience
    )
    if path is not None:
        write_atomically(path, save_state_dict({k: v.cpu() for k, v in encoder.state_dict().items()}))
        _logger.info("Gene autoencoder cached: %s", path)
    return encoder


@register(
    "aeGeneExpression",
    description="Gene-expression autoencoder embeddings fitted on the unique training samples.",
)
class AEGeneExpressionFeaturizer(Featurizer):
    """Encode the genes of a gene list into the 512-dimensional DIPK latent space."""

    sides: ClassVar[frozenset[Side]] = frozenset({"sample"})
    requires_early_stopping: ClassVar[bool] = True
    input_views: ClassVar[tuple[str, ...]] = ("gene_expression",)

    def __init__(
        self,
        *,
        view: str = "gene_expression",
        gene_list: str | None = DEFAULT_GENE_LIST,
        epochs_autoencoder: int = 2000,
        lr_autoencoder: float = 1e-4,
        patience_autoencoder: int = 50,
        batch_size_autoencoder: int = 1024,
        autoencoder_cache_dir: str | None = None,
    ) -> None:
        """Store the gene list, the autoencoder settings and the cache location.

        :param view: Name of the gene-expression view to encode.
        :param gene_list: Name of a shipped gene list to keep, in list order; the view must hold every gene of
            it. ``None`` keeps the whole view.
        :param epochs_autoencoder: Maximum number of autoencoder training epochs.
        :param lr_autoencoder: Learning rate of the autoencoder's Adam optimizer.
        :param patience_autoencoder: Epochs without improvement before the autoencoder stops.
        :param batch_size_autoencoder: Mini-batch size of the autoencoder.
        :param autoencoder_cache_dir: Directory caching trained encoders; ``None`` falls back to the environment
            variable ``DREVAL_AE_CACHE`` and, when that is unset too, disables the cache.
        """
        self._view = view
        self._gene_list = gene_list
        self._epochs = int(epochs_autoencoder)
        self._lr = float(lr_autoencoder)
        self._patience = int(patience_autoencoder)
        self._batch_size = int(batch_size_autoencoder)
        self._cache_dir = autoencoder_cache_dir
        self._names: tuple[str, ...] | None = None
        self._encoder: GeneExpressionEncoder | None = None

    @classmethod
    def output_blocks(cls, config: Mapping[str, Any]) -> tuple[BlockSpec, ...]:
        """Publish under the canonical gene-expression block name, whatever view is read.

        :param config: Unused.
        :returns: The ``gene_expression`` block.
        """
        _ = config
        return (BlockSpec(_BLOCK, FeatureFormat.NUMERIC_MATRIX),)

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return the tunable autoencoder settings.

        :returns: Hyperparameter search space mapping.
        """
        return {
            "epochs_autoencoder": {"type": "int", "low": 1, "high": 2000, "default": 2000},
            "lr_autoencoder": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True, "default": 1e-4},
            "patience_autoencoder": {"type": "int", "low": 1, "high": 100, "default": 50},
            "batch_size_autoencoder": {
                "type": "categorical",
                "choices": [32, 64, 128, 256, 512, 1024],
                "default": 1024,
            },
        }

    def valid_mask(self, source: FeatureSource, ids: np.ndarray) -> np.ndarray:
        """Return which samples have a row of the bound view that is not all NaN.

        :param source: Feature source of the sample side.
        :param ids: Sample IDs to check.
        :returns: Boolean array with one entry per ID.
        """
        return all_nan_rows_mask(source, self._view, ids)

    def _preprocess(self, source: FeatureSource, matrix: np.ndarray) -> np.ndarray:
        """Apply ``log2(x + 1)`` to linear expression units; log-scale views pass unchanged.

        :param source: Feature source the matrix came from; its ``expression_unit`` decides.
        :param matrix: Gene-expression matrix restricted to the fitted genes.
        :returns: Matrix on a log scale.
        """
        if _is_linear_expression(source, self._view):
            return np.log2(np.clip(matrix, 0.0, None) + 1.0)
        return matrix

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Train (or load) the autoencoder on the sorted unique training samples, validated on the early-stopping ones.

        :param source: Feature source of the sample side.
        :param train_ids: Training sample IDs.
        :param train_pair_ids: Unused.
        :param es_pair_ids: Sample ID of every early-stopping pair; their unique samples validate the
            autoencoder.
        :returns: This featurizer, fitted.
        :raises InvalidInputError: If no training or early-stopping sample is given.
        """
        _ = train_pair_ids
        if es_pair_ids is None or len(es_pair_ids) == 0 or len(train_ids) == 0:
            msg = f"{type(self).__name__} requires a usable early-stopping holdout"
            raise InvalidInputError(msg)
        self._names = view_genes(source, self._view, self._gene_list)
        fit_ids, validation_ids = np.unique(train_ids), np.unique(es_pair_ids)
        train = self._preprocess(source, view_matrix(source, self._view, fit_ids, self._names))
        validation = self._preprocess(source, view_matrix(source, self._view, validation_ids, self._names))
        self._encoder = train_or_load_autoencoder(
            train,
            validation,
            fit_ids,
            validation_ids,
            epochs=self._epochs,
            lr=self._lr,
            patience=self._patience,
            batch_size=self._batch_size,
            gene_list=self._gene_list,
            genes=self._names,
            cache_dir=self._cache_dir,
        )
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        """Encode the rows of *ids* into latent vectors.

        :param source: Feature source of the sample side.
        :param ids: Sample IDs, in output order.
        :returns: One unnamed ``gene_expression`` block of latent vectors.
        :raises RuntimeError: If called before ``fit``.
        """
        from drevalpy.components.predictors.literature.dipk.gene_expression_encoder import encode_gene_expression

        if self._encoder is None:
            msg = "AEGeneExpressionFeaturizer must be fit before transform"
            raise RuntimeError(msg)
        if len(ids) == 0:
            matrix = np.empty((0, int(self._encoder.latent_dim)))
        else:
            raw = view_matrix(source, self._view, ids, self._names)
            matrix = encode_gene_expression(self._preprocess(source, raw), self._encoder)
        return dense_result(ids, _BLOCK, matrix, None)
