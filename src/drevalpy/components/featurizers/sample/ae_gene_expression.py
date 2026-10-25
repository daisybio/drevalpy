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

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy._utils.torch_io import load_state_dict, save_state_dict
from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers.sample._ae_cache import cache_file, resolve_cache_dir, write_atomically
from drevalpy.components.featurizers.sample.base import DenseViewSampleFeaturizer
from drevalpy.components.featurizers.sample.dipk_gene_expression import _is_linear_expression
from drevalpy.components.featurizers.sample.gene_lists import select_genes
from drevalpy.errors import InvalidInputError
from drevalpy.log import get_logger
from drevalpy.registry.sample_featurizer import register
from drevalpy.types.data.batch.feature_block import BlockSpec
from drevalpy.types.data.feature_source import FeatureSource

if TYPE_CHECKING:
    from drevalpy.components.predictors.literature.dipk.gene_expression_encoder import GeneExpressionEncoder

_logger = get_logger(__name__)

#: Default gene list: the genes of the TCGA autoencoder that the curated screens have.
DEFAULT_GENE_LIST = "tcga_autoencoder_genes_available"

#: Autoencoder settings kept in the state: state key, attribute, accepted type.
_SETTINGS = (
    ("epochs_autoencoder", "_epochs", int),
    ("lr_autoencoder", "_lr", float),
    ("patience_autoencoder", "_patience", int),
    ("batch_size_autoencoder", "_batch_size", int),
)


@register(
    "aeGeneExpression",
    description="Gene-expression autoencoder embeddings fitted on the unique training samples.",
    output_format=FeatureFormat.NUMERIC_MATRIX,
)
class AEGeneExpressionFeaturizer(DenseViewSampleFeaturizer):
    """Encode the genes of a gene list into the 512-dimensional DIPK latent space."""

    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec("gene_expression", FeatureFormat.NUMERIC_MATRIX),)
    requires_early_stopping: ClassVar[bool] = True
    input_views: ClassVar[tuple[str, ...]] = ("gene_expression",)
    requires_fit: ClassVar[bool] = True

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
        super().__init__(view=view)
        self._gene_list = gene_list
        self._epochs = int(epochs_autoencoder)
        self._lr = float(lr_autoencoder)
        self._patience = int(patience_autoencoder)
        self._batch_size = int(batch_size_autoencoder)
        self._cache_dir = autoencoder_cache_dir
        self._encoder: GeneExpressionEncoder | None = None
        self._input_dim = 0
        self._latent_dim = 512
        self._validation_ids: np.ndarray | None = None

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

        :param source: Unused.
        :param entity_ids: Unused; IDs come from *pair_expanded_ids*.
        :param pair_expanded_ids: Training entity IDs with duplicates per response pair.
        :param pair_expanded_es_ids: Early-stopping entity IDs with duplicates.
        :returns: The sorted unique training IDs.
        :raises InvalidInputError: If *pair_expanded_ids* is missing.
        """
        _ = source, entity_ids
        if pair_expanded_ids is None:
            raise InvalidInputError("aeGeneExpression needs the training sample ids of the pairs to fit on")
        self._validation_ids = None if pair_expanded_es_ids is None else np.unique(pair_expanded_es_ids)
        return np.unique(pair_expanded_ids)

    def _preprocess(self, source: FeatureSource, matrix: np.ndarray) -> np.ndarray:
        """Apply ``log2(x + 1)`` to linear expression units; log-scale views pass unchanged.

        :param source: Feature source the matrix came from; its ``expression_unit`` decides.
        :param matrix: Gene-expression matrix restricted to the fitted genes.
        :returns: Matrix on a log scale.
        """
        if _is_linear_expression(source, self._view):
            return np.log2(np.clip(matrix, 0.0, None) + 1.0)
        return matrix

    def _autoencoder_config(self) -> dict[str, object]:
        """Collect the settings that change the trained weights.

        :returns: Mapping that goes into the cache key.
        """
        return {
            "gene_list": self._gene_list,
            "epochs": self._epochs,
            "batch_size": self._batch_size,
            "patience": self._patience,
            "lr": self._lr,
        }

    def _train_or_load(
        self, train: np.ndarray, validation: np.ndarray, train_ids: np.ndarray, validation_ids: np.ndarray
    ) -> GeneExpressionEncoder:
        """Return the encoder for this fold, from the cache when possible.

        :param train: Preprocessed training matrix, one row per unique sample.
        :param validation: Preprocessed early-stopping matrix.
        :param train_ids: Sample ids of the training rows.
        :param validation_ids: Sample ids of the early-stopping rows.
        :returns: Trained encoder in evaluation mode.
        """
        from drevalpy.components.predictors.literature.dipk._autoencoder_batches import avoid_singleton_batch
        from drevalpy.components.predictors.literature.dipk.gene_expression_encoder import (
            GeneExpressionEncoder,
            train_gene_expession_autoencoder,
        )

        directory = resolve_cache_dir(self._cache_dir)
        path = None
        if directory is not None:
            genes = self._view_names or tuple(str(i) for i in range(train.shape[1]))
            path = cache_file(
                directory, self._autoencoder_config(), genes, train_ids, validation_ids, (train, validation)
            )
            if path.is_file():
                encoder = GeneExpressionEncoder(int(train.shape[1]))
                encoder.load_state_dict(load_state_dict(path.read_bytes()))
                encoder.eval()
                _logger.info("Gene autoencoder taken from cache: %s", path)
                return encoder
        train, validation = avoid_singleton_batch(train, validation, self._batch_size)
        encoder = train_gene_expession_autoencoder(
            train,
            validation,
            self._epochs,
            lr=self._lr,
            batch_size=self._batch_size,
            patience=self._patience,
        )
        if path is not None:
            write_atomically(path, save_state_dict({k: v.cpu() for k, v in encoder.state_dict().items()}))
            _logger.info("Gene autoencoder cached: %s", path)
        return encoder

    def _fit_state(self, source: FeatureSource, entity_ids: np.ndarray) -> int:
        """Train (or load) the autoencoder and return its latent width.

        :param source: Feature source providing view matrices.
        :param entity_ids: Unique training entity IDs.
        :returns: Latent dimensionality of the encoder.
        :raises InvalidInputError: If the train or early-stopping ID set is missing or empty.
        """
        validation_ids = self._validation_ids
        if validation_ids is None or len(entity_ids) == 0 or len(validation_ids) == 0:
            raise InvalidInputError("aeGeneExpression needs non-empty training and early-stopping samples")
        train = self._preprocess(source, self._raw_matrix(source, entity_ids))
        validation = self._preprocess(source, self._raw_matrix(source, validation_ids))
        self._input_dim = int(train.shape[1])
        self._encoder = self._train_or_load(train, validation, entity_ids, validation_ids)
        self._latent_dim = int(self._encoder.latent_dim)
        return self._latent_dim

    def _compute_matrix(self, source: FeatureSource, matrix: np.ndarray) -> np.ndarray:
        """Encode *matrix* into latent vectors.

        Reached only after the ``requires_fit`` gate, so the encoder exists.

        :param source: Feature source the matrix came from.
        :param matrix: Gene-expression matrix restricted to the fitted genes.
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

    def get_state(self) -> dict[str, object]:
        """Serialize the encoder weights and the settings.

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
            "epochs_autoencoder": self._epochs,
            "lr_autoencoder": self._lr,
            "patience_autoencoder": self._patience,
            "batch_size_autoencoder": self._batch_size,
        }

    def set_state(self, state: dict[str, object]) -> None:
        """Restore an encoder serialized by ``get_state``.

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
        for key, attribute, kind in _SETTINGS:
            value = state.get(key)
            if isinstance(value, kind):
                setattr(self, attribute, value)
        self._encoder = GeneExpressionEncoder(input_dim, latent_dim=latent_dim)
        self._encoder.load_state_dict(load_state_dict(blob))
        self._encoder.eval()
        self._restore_dense_state(state)
        self._is_fitted = True
