"""BPE PharmaFormer drug featurizer with proper fit/transform separation."""

from __future__ import annotations

import io
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers._dense import dense_result
from drevalpy.components.featurizers.drug._smiles_utils import get_smiles_for_entities
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.registry.components._metadata import LiteratureReference
from drevalpy.registry.featurizer import register
from drevalpy.types.data.batch.feature_block import BlockSpec, FeaturizerResult
from drevalpy.types.data.feature_source import FeatureSource, Side

if TYPE_CHECKING:
    import pandas as pd


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


_BLOCK = "bpe_smiles"

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
)
class BpePharmaformerDrugFeaturizer(Featurizer):
    """BPE PharmaFormer drug featurizer with proper fit/transform.

    Set-dependent: BPE codes are learned from the training SMILES during fit
    and applied to encode any SMILES during transform.
    """

    sides: ClassVar[frozenset[Side]] = frozenset({"drug"})
    input_views: ClassVar[tuple[str, ...]] = ("bpe_smiles",)
    source_views: ClassVar[tuple[str, ...]] = ("canonical_smiles",)

    def __init__(self, *, view: str = "bpe_smiles", num_symbols: int = 10000, max_length: int = 128) -> None:
        """Configure the BPE vocabulary and encoded sequence length.

        :param view: Name of the view the encoded tokens are published under.
        :param num_symbols: Number of BPE merge operations (vocabulary size).
        :param max_length: Maximum encoded sequence length.
        """
        self._view = view
        self._num_symbols = int(num_symbols)
        self._max_length = int(max_length)
        self._bpe: Any = None

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable hyperparameter specs.

        :returns: HP space mapping.
        """
        return {
            "num_symbols": {"type": "categorical", "choices": [5000, 10000, 20000], "default": 10000},
            "max_length": {"type": "pow2", "low": 6, "high": 8, "default": 128},
        }

    @classmethod
    def output_blocks(cls, config: Mapping[str, Any]) -> tuple[BlockSpec, ...]:
        """Return the single ``bpe_smiles`` block.

        :param config: Featurizer config; unused.
        :returns: One numeric block spec.
        """
        return (BlockSpec(_BLOCK, FeatureFormat.NUMERIC_MATRIX),)

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Learn BPE codes from training SMILES.

        :param source: Drug feature source.
        :param train_ids: Usable training drug IDs.
        :param train_pair_ids: Unused.
        :param es_pair_ids: Unused.
        :returns: This featurizer, fitted.
        :raises ValueError: If the source has no SMILES for the training drugs.
        """
        smiles = get_smiles_for_entities(source, train_ids)
        if smiles is None:
            msg = "Cannot learn BPE codes: no SMILES available."
            raise ValueError(msg)
        self._bpe = self._learn_bpe(smiles)
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        """Encode SMILES with the learned BPE codes into one ``bpe_smiles`` block.

        Drugs without a SMILES get a zero row.

        :param source: Drug feature source.
        :param ids: Drug IDs to encode, in output order.
        :returns: Result with a ``(n_drugs, max_length)`` block.
        :raises RuntimeError: If called before the featurizer is fitted.
        :raises ValueError: If the source has no SMILES for the requested drugs.
        """
        if self._bpe is None:
            msg = "BpePharmaformerDrugFeaturizer must be fit before transform"
            raise RuntimeError(msg)
        if len(ids) == 0:
            return dense_result(ids, _BLOCK, np.empty((0, self._max_length), dtype=np.float32), None)
        smiles = get_smiles_for_entities(source, ids)
        if smiles is None:
            msg = "Cannot encode BPE: no SMILES available."
            raise ValueError(msg)
        return dense_result(ids, _BLOCK, self._apply_bpe(smiles, ids), None)

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

    def _apply_bpe(self, smiles_series: pd.Series, ids: np.ndarray) -> np.ndarray:
        """Apply stored BPE codes to encode SMILES strings.

        :param smiles_series: Series of SMILES indexed by entity IDs.
        :param ids: Drug identifiers.
        :returns: BPE token matrix of shape (n_drugs, max_length).
        """
        results = np.zeros((len(ids), self._max_length), dtype=np.int32)
        for i, drug_id in enumerate(ids):
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
