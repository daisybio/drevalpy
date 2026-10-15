"""Random Forest on DIPK's two feature building blocks instead of DIPK's neural network.

DIPK predicts drug response with a neural network on top of three learned inputs: an
autoencoder embedding of gene expression, node-level MolGNet embeddings of the drug molecule,
and BIONIC network features. ``RandomForestAEMolGNet`` keeps the first two of those inputs,
drops the BIONIC network features entirely, and replaces the network with a plain Random Forest.

The point is an ablation that the benchmark could not answer before: DIPK improves over a
Random Forest baseline, but it changes *both* the features and the predictor at the same time.
This model separates the two: same predictor as the baseline, DIPK's features instead of raw
gene expression and Morgan fingerprints. On CTRPv2 in leave-cell-line-out it beats that
baseline over ten folds: 0.3372 against 0.3155 normalised Pearson correlation, winning in 9 of
the 10 folds, p = 0.027 in a paired two-sided exact Wilcoxon signed-rank test over the ten
per-fold values. A measurable part of the gap is therefore attributable to the features alone,
with no neural predictor involved. "Normalised" means normalised against the run's
``NaiveMeanEffectsPredictor``, where 0 is "no better than the naive mean-effects model"; the
numbers are not raw correlation coefficients. The configuration behind them is not identical
to the shipped defaults -- see `Where the published numbers come from`_ below.

What is different from ``DIPK``
-------------------------------

============================  ==========================================  ==================================
Aspect                        ``DIPK``                                    ``RandomForestAEMolGNet``
============================  ==========================================  ==================================
Predictor                     attention network + fully connected head    ``RandomForestRegressor``
Drug features                 node-level MolGNet, pooled by attention     node-level MolGNet, mean+max pooled
BIONIC network features       used as a third input                       not used
Autoencoder training samples  one row per *response*                      one row per *training cell line*
Autoencoder validation set    ``output_earlystopping`` cell lines         held-out fraction of training lines
Gene expression scale         as loaded                                   ``log2(x + 1)`` by default
============================  ==========================================  ==================================

Both models train the autoencoder from scratch inside every fold and only ever fit it on
training cell lines, so neither leaks test expression into the encoder.

Design decisions
----------------

1. **MolGNet pooling is mean+max.** DIPK pools the node embeddings with attention inside its
   network. A Random Forest needs a fixed-length vector, so the node matrix
   (``n_atoms x 768``) is reduced to ``1536`` features by concatenating the per-dimension mean
   and maximum. ``molgnet_pooling`` switches to ``mean`` or ``max`` alone (``768`` features).

2. **The autoencoder sees unique cell lines, not response rows.** DIPK feeds its autoencoder the
   expression matrix indexed by ``output.cell_line_ids``, which repeats every cell line once per
   measured drug. Since the autoencoder is unsupervised over cell lines, that only weights cell
   lines by how many drugs were screened on them and multiplies the number of gradient steps.
   Training on the unique training cell lines keeps the objective identical while cutting the
   input matrix by roughly two orders of magnitude; ``batch_size_autoencoder`` defaults to ``32``
   instead of DIPK's ``1024`` so that the number of gradient steps stays comparable.

3. **Validation for early stopping is a held-out fraction of training cell lines.** sklearn models
   receive no ``output_earlystopping`` set, which is where DIPK takes its autoencoder validation
   data from. The split is deterministic given ``autoencoder_seed``.

4. **Gene expression is log2-transformed before encoding.** CTRPv2 gene expression is linear TPM.
   On a linear scale the highest-expressed genes dominate the variance, and a mean-squared-error
   autoencoder then spends its capacity reconstructing housekeeping genes. Set
   ``autoencoder_log2_transform`` to ``false`` for data that is already on a log scale.

5. **Encoder caching is opt-in.** The encoder depends only on the training cell lines, the gene
   list and the autoencoder hyperparameters -- not on the drug features. When several models share
   those, ``autoencoder_cache_dir`` (or the ``DREVAL_AE_CACHE`` environment variable) lets them
   train the encoder once per fold and reuse it. Beyond saving time this makes such models
   pairwise comparable, because an identical encoder rules out the encoder as a source of
   differences. Caching is disabled unless one of the two is set.

The four autoencoder hyperparameters that DIPK also has keep DIPK's names
(``epochs_autoencoder``, ``lr_autoencoder``, ``patience_autoencoder``,
``batch_size_autoencoder``); the ones that only exist here are prefixed ``autoencoder_``.

.. _Where the published numbers come from:

Where the published numbers come from
-------------------------------------

The comparison quoted at the top was run on CTRPv2 in leave-cell-line-out over ten folds, with
the ``RandomForest`` baseline exactly as shipped. Three of the four autoencoder settings of that
run are the defaults here (``epochs_autoencoder`` 2000, ``batch_size_autoencoder`` 32,
``patience_autoencoder`` 50). The gene list is not: the run used a locally built list of 11,883
genes, which is not part of the metadata bundle drevalpy downloads, so the default is
``gene_expression_intersection`` (2,270 genes) -- DIPK's default, and available to everyone.
Running the defaults therefore encodes a gene space about five times smaller than the one behind
the quoted p-value, and the numbers will not reproduce exactly.

One more asymmetry of that run, in the baseline's favour: this model was trained on ``drug_mean``
residuals while the baseline was not, because ``drug_mean`` costs the bare Random Forest a large
amount of performance (0.3106 down to 0.2495 normalised Pearson in an earlier run), so the
baseline competed in its stronger configuration.
"""

import hashlib
import json
import os
from typing import cast

import numpy as np
import pandas as pd
import torch

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

from ..DIPK.gene_expression_encoder import (
    GeneExpressionEncoder,
    encode_gene_expression,
    train_gene_expession_autoencoder,
)
from .sklearn_models import RandomForest, SklearnModel

#: View name of the pooled MolGNet embedding. Deliberately different from DIPK's node-level
#: ``molgnet_features`` so that the two cannot be confused in a FeatureDataset.
MOLGNET_POOLED_VIEW = "molgnet_pooled"

#: File the fold encoder is written to by :meth:`AutoencodedGeneExpressionMixin.save`.
ENCODER_FILE = "gene_encoder.pt"


def pool_molgnet(nodes: np.ndarray, pooling: str = "mean_max") -> np.ndarray:
    """Aggregate a node-level MolGNet matrix into one fixed-length vector.

    :param nodes: node embeddings of a single molecule, shape ``(n_atoms, n_dim)``
    :param pooling: ``"mean"``, ``"max"`` or ``"mean_max"`` (concatenation of both)
    :returns: pooled vector with ``n_dim`` entries, or ``2 * n_dim`` for ``"mean_max"``
    :raises ValueError: if ``pooling`` is not one of the three supported values
    """
    if pooling == "mean":
        return nodes.mean(axis=0)
    if pooling == "max":
        return nodes.max(axis=0)
    if pooling == "mean_max":
        return np.concatenate((nodes.mean(axis=0), nodes.max(axis=0)))
    raise ValueError(f"Unknown MolGNet pooling: {pooling!r}. Expected 'mean', 'max' or 'mean_max'.")


def molgnet_feature_names(n_dim: int, pooling: str) -> list[str]:
    """Build the feature names of a pooled MolGNet view.

    :param n_dim: number of pooled features
    :param pooling: pooling variant the names belong to
    :returns: one name per pooled feature
    """
    if pooling == "mean_max":
        half = n_dim // 2
        return [f"molgnet_mean_{i}" for i in range(half)] + [f"molgnet_max_{i}" for i in range(half)]
    return [f"molgnet_{pooling}_{i}" for i in range(n_dim)]


class MolGNetPooledMixin(SklearnModel):
    """Mixin that replaces the drug fingerprints with a pooled MolGNet embedding.

    Reads the same ``DIPK_features/Drugs/MolGNet_<drug>.csv`` files that :class:`DIPK` uses and
    pools them over the atoms, because a Random Forest cannot consume a variable-length node
    matrix.
    """

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Load all MolGNet node matrices of a dataset and pool them over the atoms.

        :param data_path: path to the data directory
        :param dataset_name: name of the dataset
        :returns: FeatureDataset with the single view ``molgnet_pooled``
        :raises FileNotFoundError: if the dataset has no ``DIPK_features/Drugs`` directory
        """
        pooling = self.hyperparameters.get("molgnet_pooling", "mean_max")
        drug_path = os.path.join(data_path, dataset_name, "DIPK_features", "Drugs")
        if not os.path.isdir(drug_path):
            raise FileNotFoundError(
                f"MolGNet features not found: {drug_path}. "
                f"{self.get_model_name()} needs the DIPK feature files of the dataset."
            )

        features: dict[str, dict[str, np.ndarray]] = {}
        n_dim = 0
        for file in sorted(os.listdir(drug_path)):
            if not (file.startswith("MolGNet_") and file.endswith(".csv")):
                continue
            drug = file[len("MolGNet_") : -len(".csv")]  # noqa: E203
            nodes = pd.read_csv(os.path.join(drug_path, file), index_col=0, sep="\t").to_numpy(dtype=np.float32)
            vector = pool_molgnet(nodes, pooling).astype(np.float32)
            n_dim = len(vector)
            features[drug] = {MOLGNET_POOLED_VIEW: vector}

        if not features:
            raise FileNotFoundError(f"No MolGNet_*.csv files in {drug_path}.")

        names = molgnet_feature_names(n_dim, pooling)
        return FeatureDataset(features=features, meta_info={MOLGNET_POOLED_VIEW: np.array(names)})


class AutoencodedGeneExpressionMixin(SklearnModel):
    """Mixin that replaces raw gene expression with the embedding of a per-fold autoencoder.

    The autoencoder is :class:`GeneExpressionEncoder`, the same one DIPK uses. It is trained from
    scratch in every fold and only on training cell lines; test cell lines are merely transformed
    with the finished encoder.
    """

    #: Encoder of the current fold, set by :meth:`train` and by :meth:`load`.
    _encoder: GeneExpressionEncoder | None = None
    #: Number of genes the encoder was built for.
    _ae_input_dim: int | None = None

    def build_model(self, hyperparameters: dict) -> None:
        """Build the underlying model and read the autoencoder configuration.

        Recognised autoencoder hyperparameters, all optional:
        ``epochs_autoencoder`` (2000), ``batch_size_autoencoder`` (32),
        ``patience_autoencoder`` (50), ``lr_autoencoder`` (1e-4),
        ``autoencoder_log2_transform`` (true), ``autoencoder_validation_fraction`` (0.1),
        ``autoencoder_seed`` (42), ``autoencoder_cache_dir`` (unset).

        :param hyperparameters: hyperparameters of the model
        """
        super().build_model(hyperparameters)
        hpams = self.hyperparameters
        self._ae_log2 = bool(hpams.get("autoencoder_log2_transform", True))
        self._ae_epochs = int(hpams.get("epochs_autoencoder", 2000))
        self._ae_batch_size = int(hpams.get("batch_size_autoencoder", 32))
        self._ae_patience = int(hpams.get("patience_autoencoder", 50))
        self._ae_lr = float(hpams.get("lr_autoencoder", 1e-4))
        self._ae_val_fraction = float(hpams.get("autoencoder_validation_fraction", 0.1))
        self._ae_seed = int(hpams.get("autoencoder_seed", 42))
        self._ae_cache_dir = hpams.get("autoencoder_cache_dir") or os.environ.get("DREVAL_AE_CACHE") or None
        self._encoder: GeneExpressionEncoder | None = None
        self._ae_input_dim: int | None = hpams.get("gene_encoder_input_dim")

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Load gene expression and, unless switched off, transform it to ``log2(x + 1)``.

        :param data_path: path to the data directory
        :param dataset_name: name of the dataset
        :returns: FeatureDataset with the (optionally log2-transformed) gene expression
        """
        cell_line_input = super().load_cell_line_features(data_path, dataset_name)
        if self._ae_log2:
            cell_line_input.apply(lambda x: np.log2(np.clip(x, 0.0, None) + 1.0), view="gene_expression")
        return cell_line_input

    def _autoencoder_config(self) -> dict:
        """Collect everything that defines the identity of an encoder.

        :returns: the configuration that goes into the cache key
        """
        return {
            "gene_list": self.gene_list,
            "log2": self._ae_log2,
            "epochs": self._ae_epochs,
            "batch_size": self._ae_batch_size,
            "patience": self._ae_patience,
            "lr": self._ae_lr,
            "val_fraction": self._ae_val_fraction,
            "seed": self._ae_seed,
        }

    def _cache_path(self, train_ids: np.ndarray, input_dim: int, genes: list[str]) -> str | None:
        """Build the cache file path for these training cell lines, config and gene *order*.

        The gene order is part of the key on purpose. Two gene lists with the same members but a
        different column order produce different encoders, and feeding a cached encoder genes in
        the wrong order would not raise -- it would silently return wrong numbers.

        :param train_ids: training cell line identifiers of this fold
        :param input_dim: number of genes going into the encoder
        :param genes: gene symbols in exactly the order they are passed to the encoder
        :returns: path of the cache file, or ``None`` if caching is not configured
        """
        if not self._ae_cache_dir:
            return None
        payload = json.dumps(
            {
                "config": self._autoencoder_config(),
                "input_dim": int(input_dim),
                "cell_lines": sorted(map(str, train_ids)),
                "gene_order": hashlib.sha256("\n".join(map(str, genes)).encode()).hexdigest(),
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
        return os.path.join(self._ae_cache_dir, f"gene_ae_{digest}.pt")

    def _fit_or_load_encoder(self, train_ids: np.ndarray, cell_line_input: FeatureDataset) -> GeneExpressionEncoder:
        """Get the encoder for this fold, from the cache if possible, otherwise by training it.

        :param train_ids: unique training cell lines of this fold
        :param cell_line_input: cell line features, still unencoded
        :returns: trained encoder
        """
        matrix = cell_line_input.get_feature_matrix(view="gene_expression", identifiers=train_ids)
        self._ae_input_dim = int(matrix.shape[1])

        gene_order = [str(gene) for gene in cell_line_input.meta_info["gene_expression"]]
        path = self._cache_path(train_ids, self._ae_input_dim, gene_order)
        if path is not None and os.path.exists(path):
            encoder = self._encoder_from_state(torch.load(path, map_location="cpu", weights_only=True))
            print(f"[AE] encoder taken from cache: {path} ({len(train_ids)} cell lines, {self._ae_input_dim} genes)")
            return encoder

        rng = np.random.default_rng(self._ae_seed)
        order = rng.permutation(len(matrix))
        n_val = max(1, int(round(self._ae_val_fraction * len(matrix))))
        # The encoder uses BatchNorm1d, which cannot normalise a batch of one sample while
        # training. A trailing mini batch of exactly one row would therefore abort the fold, so
        # one row moves to the validation set instead. Training on unique cell lines makes this
        # reachable; DIPK never hits it because it trains on the far longer response matrix.
        if (len(matrix) - n_val) % self._ae_batch_size == 1:
            n_val += 1
        val_matrix, train_matrix = matrix[order[:n_val]], matrix[order[n_val:]]
        print(
            f"[AE] training gene autoencoder from scratch: {len(train_matrix)} cell lines "
            f"(+{len(val_matrix)} validation), {self._ae_input_dim} genes, "
            f"batch_size={self._ae_batch_size}, up to {self._ae_epochs} epochs, patience={self._ae_patience}"
        )
        encoder = train_gene_expession_autoencoder(
            train_matrix,
            val_matrix,
            epochs_autoencoder=self._ae_epochs,
            lr=self._ae_lr,
            patience=self._ae_patience,
            batch_size=self._ae_batch_size,
        )
        if path is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self._encoder_state(encoder), path)
            print(f"[AE] encoder cached: {path}")
        return encoder

    def _encoder_state(self, encoder: GeneExpressionEncoder) -> dict:
        """Build the serialisable state of an encoder.

        :param encoder: encoder to serialise
        :returns: dictionary that :meth:`_encoder_from_state` can read back
        """
        return {
            "encoder": {key: value.cpu() for key, value in encoder.state_dict().items()},
            "input_dim": self._ae_input_dim,
            "config": self._autoencoder_config(),
        }

    @staticmethod
    def _encoder_from_state(state: dict) -> GeneExpressionEncoder:
        """Rebuild an encoder from a state saved by :meth:`_encoder_state`.

        :param state: state dictionary
        :returns: encoder in evaluation mode
        """
        encoder = GeneExpressionEncoder(state["input_dim"])
        encoder.load_state_dict(state["encoder"])
        encoder.eval()
        return encoder

    def _encode(self, cell_line_input: FeatureDataset) -> FeatureDataset:
        """Encode the ``gene_expression`` view in place with the fold encoder.

        Encoding in place is safe because ``experiment.py`` hands every model call a deep copy of
        the features. A dataset that is already encoded is recognised and left alone.

        :param cell_line_input: cell line features
        :returns: the same FeatureDataset, now encoded
        :raises ValueError: if the features have a different number of genes than the encoder was
            built for, which would mean a gene space other than the one it was trained on
        """
        first = next(iter(cell_line_input.features.values()))["gene_expression"]
        if self._encoder is not None and len(first) == self._encoder.latent_dim:
            return cell_line_input
        # An encoder is only ever valid for the gene space it was trained on. The matrix
        # multiplication would raise anyway, but a named error is worth more than a shape error,
        # and silently encoding a permuted or differently sized gene space would be worse still.
        if self._ae_input_dim is not None and len(first) != self._ae_input_dim:
            raise ValueError(
                f"{self.get_model_name()}: the gene encoder was built for {self._ae_input_dim} genes "
                f"but the features have {len(first)}. The gene_list must be the same one the encoder "
                f"was trained with."
            )
        cell_line_input.apply(
            lambda x: encode_gene_expression(x, self._encoder),  # type: ignore[arg-type]
            view="gene_expression",
        )
        return cell_line_input

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """Fit the autoencoder on the training cell lines, encode, then train as usual.

        :param output: training response data
        :param cell_line_input: cell line features
        :param drug_input: drug features
        :param output_earlystopping: unused, sklearn models do not early stop
        :param model_checkpoint_dir: unused
        """
        if len(output) > 0:
            train_ids = np.unique(np.asarray(output.cell_line_ids).astype(str))
            self._encoder = self._fit_or_load_encoder(train_ids, cell_line_input)
            self.hyperparameters["gene_encoder_input_dim"] = self._ae_input_dim
            cell_line_input = self._encode(cell_line_input)
        super().train(
            output=output,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
            output_earlystopping=output_earlystopping,
            model_checkpoint_dir=model_checkpoint_dir,
        )

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """Encode the gene expression with the fold encoder, then predict as usual.

        :param cell_line_ids: cell line identifiers
        :param drug_ids: drug identifiers
        :param cell_line_input: cell line features
        :param drug_input: drug features
        :returns: predicted responses
        """
        if self._encoder is not None:
            cell_line_input = self._encode(cell_line_input)
        return super().predict(
            cell_line_ids=cell_line_ids,
            drug_ids=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )

    def save(self, directory: str) -> None:
        """Save the usual artefacts plus the fold encoder, without which the model cannot predict.

        :param directory: target directory
        """
        super().save(directory)
        if self._encoder is not None:
            torch.save(self._encoder_state(self._encoder), os.path.join(directory, ENCODER_FILE))

    @classmethod
    def load(cls, directory: str) -> SklearnModel:
        """Load the usual artefacts plus the gene encoder.

        :param directory: directory holding the saved artefacts
        :returns: restored model instance
        """
        instance = cast("AutoencodedGeneExpressionMixin", super().load(directory))
        path = os.path.join(directory, ENCODER_FILE)
        if os.path.exists(path):
            state = torch.load(path, map_location="cpu", weights_only=True)
            instance._encoder = cls._encoder_from_state(state)
            instance._ae_input_dim = state["input_dim"]
        return instance


class RandomForestAEMolGNet(AutoencodedGeneExpressionMixin, MolGNetPooledMixin, RandomForest):
    """Random Forest on an autoencoder embedding of gene expression and pooled MolGNet features.

    Uses DIPK's two main feature building blocks with a Random Forest instead of DIPK's neural
    network. See the module docstring for the full comparison with :class:`DIPK`.
    """

    #: Gene list the autoencoder input is restricted to. Same default as ``DIPK`` so that both
    #: models encode the same gene space. Overridable via the ``gene_list`` hyperparameter.
    gene_list: str | None = "gene_expression_intersection"

    @classmethod
    def get_model_name(cls) -> str:
        """Return the model name.

        :returns: RandomForestAEMolGNet
        """
        return "RandomForestAEMolGNet"
