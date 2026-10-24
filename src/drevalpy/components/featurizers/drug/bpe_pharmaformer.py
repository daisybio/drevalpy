"""BPE PharmaFormer drug featurizer with proper fit/transform separation."""

from __future__ import annotations

import io
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.drug._smiles_utils import get_smiles_for_entities
from drevalpy.components.featurizers.drug.base import DrugFeaturizer
from drevalpy.log import get_logger
from drevalpy.registry.drug_featurizer import register
from drevalpy.types.data.batch.feature_block import BlockSpec, FeaturizerResult, numeric_feature_block
from drevalpy.types.data.feature_source import FeatureSource
from drevalpy.types.enums.literature_reference import LiteratureReference

if TYPE_CHECKING:
    import pandas as pd

_logger = get_logger(__name__)


@contextmanager
def _silenced_learn_bpe_progress() -> Iterator[None]:
    """Replace the progress bar ``subword_nmt.learn_bpe`` wraps its merge loop in by the bare iterable.

    ``learn_bpe`` looks ``tqdm`` up in its own module namespace and offers no switch for it.
    The original binding is restored on exit.

    :yields: Control, with the progress bar disabled.
    """
    from subword_nmt import learn_bpe as learn_bpe_module

    original = learn_bpe_module.tqdm
    learn_bpe_module.tqdm = lambda iterable, *args, **kwargs: iterable
    try:
        yield
    finally:
        learn_bpe_module.tqdm = original


_BPE_PHARMAFORMER_REFERENCE = LiteratureReference(
    repo_url="https://github.com/zhouyuru1205/PharmaFormer",
    citation_doi="10.1038/s41698-025-01082-6",
    deviations=(
        "Set-dependent featurizer: BPE codes are learned from training SMILES "
        "at fit time and applied to any SMILES at transform time."
    ),
)


@register(
    "bpePharmaformer",
    description="BPE PharmaFormer token rows computed via fit/transform (set-dependent).",
    reference=_BPE_PHARMAFORMER_REFERENCE,
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class BpePharmaformerDrugFeaturizer(DrugFeaturizer):
    """BPE PharmaFormer drug featurizer with proper fit/transform.

    Set-dependent: BPE codes are learned from the training SMILES during fit
    and applied to encode any SMILES during transform.
    """

    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec("bpe_smiles", FeatureFormat.NUMERIC_MATRIX),)
    storage_key: ClassVar[str] = "bpe_smiles"
    input_views: ClassVar[tuple[str, ...]] = ("bpe_smiles",)
    source_views: ClassVar[tuple[str, ...]] = ("canonical_smiles",)
    precompute: ClassVar[bool] = False

    def __init__(self, *, view: str = "bpe_smiles", num_symbols: int = 10000, max_length: int = 128) -> None:
        """Configure the BPE vocabulary and encoded sequence length.

        :param view: Name of the view the encoded tokens are published under.
        :param num_symbols: Number of BPE merge operations (vocabulary size).
        :param max_length: Maximum encoded sequence length.
        """
        self._view = view
        self._num_symbols = int(num_symbols)
        self._max_length = int(max_length)
        self._output_dim = self._max_length
        self._bpe = None

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable hyperparameter specs.

        :returns: HP space mapping.
        """
        return {
            "num_symbols": {"type": "categorical", "choices": [5000, 10000, 20000], "default": 10000},
            "max_length": {"type": "pow2", "low": 6, "high": 8, "default": 128},
        }

    def _fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> BpePharmaformerDrugFeaturizer:
        """Learn BPE codes from training SMILES.

        :param source: Feature source providing drug views.
        :param entity_ids: Training drug identifiers.
        :param pair_expanded_ids: Unused.
        :param pair_expanded_es_ids: Unused.
        :returns: Fitted featurizer instance.
        :raises ValueError: If the source has no SMILES for the training drugs.
        """
        _ = pair_expanded_ids, pair_expanded_es_ids
        ids = entity_ids if entity_ids is not None else source.identifiers

        smiles = get_smiles_for_entities(source, ids)
        if smiles is None:
            msg = "Cannot learn BPE codes: no SMILES available."
            raise ValueError(msg)

        self._bpe = self._learn_bpe(smiles)
        self._output_dim = self._max_length
        return self

    def _feature_matrix(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Apply learned BPE codes to encode SMILES.

        :param source: Feature source providing drug views.
        :param entity_ids: Drug identifiers to transform.
        :returns: BPE token matrix of shape (n_drugs, max_length).
        :raises RuntimeError: If called before the featurizer is fitted.
        :raises ValueError: If the source has no SMILES for the requested drugs.
        """
        if self._bpe is None:
            msg = "BpePharmaformerDrugFeaturizer must be fit before transform"
            raise RuntimeError(msg)

        if len(entity_ids) == 0:
            return np.empty((0, self.output_dim), dtype=np.float32)
        smiles = get_smiles_for_entities(source, entity_ids)
        if smiles is None:
            msg = "Cannot encode BPE: no SMILES available."
            raise ValueError(msg)
        return self._apply_bpe(smiles, entity_ids).astype(np.float32)

    def _transform_features(self, source: FeatureSource, entity_ids: np.ndarray) -> FeaturizerResult:
        """Wrap the BPE token matrix in a single ``bpe_smiles`` block.

        :param source: Feature source providing drug views.
        :param entity_ids: Drug identifiers to encode.
        :returns: Result with one numeric block.
        """
        matrix = self._feature_matrix(source, entity_ids)
        return FeaturizerResult(
            entity_ids,
            {
                "bpe_smiles": numeric_feature_block(
                    matrix,
                    feature_names=None,
                )
            },
            ("bpe_smiles",),
        )

    @property
    def output_dim(self) -> int:
        """Return the encoded sequence length (``max_length``)."""
        return self._output_dim

    def get_state(self) -> dict[str, object]:
        """Persist learned BPE merges and sequence width.

        :returns: Fitted tokenizer state, or an empty mapping before fitting.
        """
        if self._bpe is None:
            return {}
        return {"bpe": self._bpe, "max_length": self._max_length, "output_dim": self._output_dim}

    def set_state(self, state: dict[str, object]) -> None:
        """Restore the fitted tokenizer used during training.

        :param state: Mapping produced by ``get_state``.
        """
        self._bpe = state.get("bpe")
        self._max_length = int(state.get("max_length", self._max_length))
        self._output_dim = int(state.get("output_dim", 0))

    def _learn_bpe(self, smiles_series: pd.Series) -> object:
        """Learn BPE codes from a set of SMILES strings.

        :param smiles_series: Series of SMILES indexed by entity IDs.
        :returns: Fitted BPE object.
        :raises ImportError: If ``subword-nmt`` is not installed.
        """
        try:
            from subword_nmt.apply_bpe import BPE
            from subword_nmt.learn_bpe import learn_bpe
        except ImportError as err:
            msg = "subword-nmt is required for BPE computation: pip install subword-nmt"
            raise ImportError(msg) from err

        all_smiles = smiles_series.dropna()

        smiles_lines = io.StringIO("".join(f"{smi}\n" for smi in all_smiles))
        bpe_codes = io.StringIO()
        with _silenced_learn_bpe_progress():
            learn_bpe(smiles_lines, bpe_codes, num_symbols=self._num_symbols, verbose=False)
        return BPE(bpe_codes)

    def _apply_bpe(self, smiles_series: pd.Series, entity_ids: np.ndarray) -> np.ndarray:
        """Apply stored BPE codes to encode SMILES strings.

        :param smiles_series: Series of SMILES indexed by entity IDs.
        :param entity_ids: Drug identifiers.
        :returns: BPE token matrix of shape (n_drugs, max_length).
        """
        results = np.zeros((len(entity_ids), self._max_length), dtype=np.int32)
        for i, drug_id in enumerate(entity_ids):
            smi = smiles_series.get(drug_id)
            if smi and isinstance(smi, str):
                bpe_processed = self._bpe.process_line(smi)
                encoded = [ord(char) for char in bpe_processed]
                if len(encoded) > self._max_length:
                    encoded = encoded[: self._max_length]
                else:
                    encoded = list(np.pad(encoded, (0, self._max_length - len(encoded)), "constant"))
                results[i] = encoded
        return results.astype(np.float32)
