"""Concatenate datasets of different studies into one ordinary dataset."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

from drevalpy.errors import InvalidInputError
from drevalpy.log import get_logger
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.modalities import public_omics_name

from ._concat_stack import stack_response, stack_view
from ._concat_tables import STUDY_SEPARATOR, listed, prefixed, stack_obs
from ._concat_uns import merge_uns

if TYPE_CHECKING:
    import anndata as ad
    import pandas as pd

logger = get_logger(__name__)

#: Precomputed featurizer variants live in ``response.obsm``/``varm`` plus these ``uns`` keys.
_VARIANT_KEYS = frozenset({"sample_featurizer_variants", "drug_featurizer_variants"})
_OWN_UNS_KEYS = frozenset({"dataset_name", "randomization", "randomization_provenance"}) | _VARIANT_KEYS
_NEW_OBS_COLUMNS = ("study", "source_sample_id")


def _check_names(datasets: Sequence[Dataset], name: str | None) -> list[str]:
    """Return the input names, which must be unique, non-empty and free of the row-ID separator."""
    names = [dataset.name for dataset in datasets]
    duplicated = sorted({item for item in names if names.count(item) > 1})
    if duplicated:
        raise InvalidInputError(f"Dataset names must be unique to tell the studies apart, got duplicates: {duplicated}")
    bad = [item for item in names if not item or STUDY_SEPARATOR in item]
    if bad or name == "":
        raise InvalidInputError(
            f"Dataset names must be non-empty and must not contain {STUDY_SEPARATOR!r}: {bad or [name]}"
        )
    return names


def _check_inputs(datasets: Sequence[Dataset], join: str, name: str | None) -> list[str]:
    """Validate the arguments of :func:`concat` and return the input names."""
    if join not in ("exact", "inner"):
        raise InvalidInputError(f"join must be 'exact' or 'inner', got {join!r}")
    if len(datasets) < 2:
        raise InvalidInputError(f"concat needs at least two datasets, got {len(datasets)}")
    names = _check_names(datasets, name)
    randomized = [dataset.name for dataset in datasets if dataset.randomization is not None]
    if randomized:
        raise InvalidInputError(f"Randomized datasets cannot be concatenated: {randomized}")
    already = [dataset.name for dataset in datasets if set(_NEW_OBS_COLUMNS) & set(dataset.mdata.obs.columns)]
    if already:
        raise InvalidInputError(f"Datasets {already} already have a 'study' or 'source_sample_id' column in obs")
    return names


def _warn_not_carried(datasets: Sequence[Dataset]) -> None:
    """Warn about stored featurizer variants and embeddings, which concat does not carry over."""
    for dataset in datasets:
        response = dataset.response
        stored = [*response.obsm.keys(), *response.varm.keys(), *(_VARIANT_KEYS & set(dataset.mdata.uns))]
        if stored:
            logger.warning(
                "concat does not carry precomputed featurizer variants or embeddings; dropped from %s: %s",
                dataset.name,
                listed(stored),
            )


def _stack_obs(datasets: Sequence[Dataset], names: Sequence[str]) -> pd.DataFrame:
    """Stack the global ``obs``: every input's rows, prefixed, plus ``study`` and ``source_sample_id``."""
    import pandas as pd

    frames = []
    for dataset, study in zip(datasets, names, strict=True):
        obs = dataset.mdata.obs
        frame = obs.set_axis(prefixed(study, obs.index), axis=0)
        frame["study"] = study
        frame["source_sample_id"] = obs.index.astype(str).to_numpy()
        frames.append(frame)
    obs = stack_obs(frames)
    obs["study"] = pd.Categorical(obs["study"], categories=list(names))
    return obs


def _groups(
    datasets: Sequence[Dataset], wanted: Callable[[str], bool]
) -> dict[str, tuple[str, list[ad.AnnData | None]]]:
    """Group modalities across inputs by public name: ``name -> (key to store under, one per input)``."""
    groups: dict[str, tuple[str, list[ad.AnnData | None]]] = {}
    for position, dataset in enumerate(datasets):
        for key, adata in dataset.mdata.mod.items():
            if wanted(key):
                _, slots = groups.setdefault(public_omics_name(key), (key, [None] * len(datasets)))
                slots[position] = adata
    return groups


def _is_view(key: str) -> bool:
    return key != "response" and not key.startswith("response_")


def _is_extra_response(key: str) -> bool:
    return key.startswith("response_")


def _missing_in(slots: Sequence[ad.AnnData | None], names: Sequence[str]) -> list[str]:
    return [name for name, adata in zip(names, slots, strict=True) if adata is None]


def _stack_all_views(
    datasets: Sequence[Dataset], names: Sequence[str], join: Literal["exact", "inner"]
) -> dict[str, ad.AnnData]:
    """Stack the sample views and the extra response-shaped modalities that every input has."""
    stacked: dict[str, ad.AnnData] = {}
    for key, slots in _groups(datasets, _is_view).values():
        lacking = _missing_in(slots, names)
        if lacking and join == "exact":
            raise InvalidInputError(
                f"View '{key}' is missing in {lacking}; pass join='inner' to drop views that not every dataset has"
            )
        if lacking:
            logger.warning("concat dropped view '%s', missing in %s", key, lacking)
        else:
            stacked[key] = stack_view(key, slots, names, join)
    for key, slots in _groups(datasets, _is_extra_response).values():
        lacking = _missing_in(slots, names)
        if lacking:
            logger.warning("concat dropped modality '%s', missing in %s", key, lacking)
        else:
            stacked[key] = stack_response(slots, names, where=key, strict_uns=lambda uns_key: uns_key == "x_column")
    return stacked


def concat(
    datasets: Sequence[Dataset],
    *,
    join: Literal["exact", "inner"] = "exact",
    name: str | None = None,
) -> Dataset:
    """Concatenate datasets, one per study, into one :class:`~drevalpy.types.data.dataset.Dataset`.

    The result is an ordinary dataset, so every function that takes one (``split``,
    ``run``, ``aggregate``) works on it unchanged:

    * Each row is one (study, sample) with the ID ``<study>::<sample_id>``, where the
      study is the input's ``name``. ``obs`` gains the columns ``study`` and
      ``source_sample_id``. A sample measured in two studies gives two rows, so no
      measurement is lost.
    * Sample views are stacked row-wise, view by view.
    * Drug columns are the union of the drug IDs. Response layers are the union, with
      NaN where a study did not measure a pair.
    * ``uns["dataset_name"]`` is *name*, or the input names joined by ``+``.

    The inputs are not modified. Precomputed featurizer variants and embeddings stored
    in ``response.obsm``/``varm`` are not carried over.

    :param datasets: At least two datasets with distinct names, none of them randomized.
    :param join: ``"exact"`` requires every input to carry the same feature names and
        units in each view they share. ``"inner"`` keeps only the feature names
        common to all inputs. A unit mismatch raises with either value.
    :param name: Name of the result. Defaults to the input names joined by ``+``,
        such as ``"GDSC2+CTRPv2"``.
    :returns: The concatenated dataset.
    :raises InvalidInputError: If fewer than two datasets are given, names are not unique, an input
        is randomized, feature names or units differ between inputs, or one drug ID
        has two different SMILES strings.
    """
    import mudata as md

    datasets = list(datasets)
    names = _check_inputs(datasets, join, name)
    _warn_not_carried(datasets)
    modalities = {"response": stack_response([dataset.response for dataset in datasets], names)}
    modalities.update(_stack_all_views(datasets, names, join))
    uns = merge_uns([dataset.mdata.uns for dataset in datasets], names=names, where="the dataset", skip=_OWN_UNS_KEYS)

    with md.set_options(pull_on_update=False):
        mdata = md.MuData(modalities)
    mdata.obs = _stack_obs(datasets, names).reindex(mdata.obs_names)
    mdata.uns.update(uns)
    mdata.uns["dataset_name"] = name if name is not None else "+".join(names)
    return Dataset(mdata, name=str(mdata.uns["dataset_name"]))
