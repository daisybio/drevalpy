"""Groupings: which group each pair of a run belongs to, by drug, sample, study or annotation.

The ``"drug"`` and ``"sample"`` groupings read a run's IDs. The ``"study"`` grouping reads the
study from the ``"<study>::"`` prefix of the sample IDs that ``drevalpy.data.concat`` writes, so
it splits a concatenated dataset into its studies; a sample ID without the prefix belongs to no
study. Any sample or drug annotation column of a dataset passed in (``tissue``, ``ncit_class``,
...) is a grouping too; a pair whose annotation is missing, empty or ``"unknown"``, or whose ID
is not in the dataset, belongs to no group. Labels of pairs without a group are NaN, see
:func:`defined_labels`. Scoring the groups is :mod:`drevalpy.evaluation._scoring`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from drevalpy.types import Dataset
    from drevalpy.types.results import RunResult

#: The groupings that need no dataset, in report order.
GROUPINGS: Final[tuple[str, ...]] = ("drug", "sample", "study")

#: A grouping of pairs: ``"drug"``, ``"sample"``, ``"study"``, or a sample/drug annotation column.
GroupBy = str

#: Grouping name -> the ``RunResult`` attribute holding that grouping's labels.
_ID_ATTRIBUTE: Final[dict[str, str]] = {"drug": "drug_ids", "sample": "sample_ids", "study": "sample_ids"}

#: Prefixes that select the axis of an annotation grouping, such as ``"sample:tissue"``.
_AXES: Final[tuple[str, ...]] = ("sample", "drug")

#: Separates the study from the source sample ID in the sample IDs of a concatenated dataset.
STUDY_SEPARATOR: Final[str] = "::"

#: The label a dataset writes for an annotation it could not resolve.
UNKNOWN_LABEL: Final[str] = "unknown"


def group_labels(run: RunResult, grouping: GroupBy, dataset: Dataset | None = None) -> np.ndarray:
    """Return the grouping labels of a run.

    Raises ``ValueError`` if ``grouping`` is not a built-in grouping or an annotation column of ``dataset``,
    is a bare name found on both axes, or needs a ``dataset`` that was not given.

    :param run: Run whose identifier arrays to read.
    :param grouping: ``"drug"``, ``"sample"`` or ``"study"``, or the name of a sample or drug annotation
        column of ``dataset``. ``"sample:<column>"`` and ``"drug:<column>"`` pick the axis explicitly.
    :param dataset: Dataset whose annotations label the pairs; needed only for an annotation ``grouping``.
    :returns: The run's ``drug_ids`` or ``sample_ids`` array. For ``"study"``, an object array
        of the sample ID prefixes before ``"::"`` (the study of a concatenated dataset), NaN
        where a sample ID has no prefix. For an annotation, an object array of the annotation of each
        pair's sample (drug) as a string, NaN where it is missing, empty, ``"unknown"`` or the ID is not
        in the dataset; see :func:`defined_labels`.
    """
    return group_labeller(grouping, dataset)(run)


def group_labeller(grouping: GroupBy, dataset: Dataset | None = None) -> Callable[[RunResult], np.ndarray]:
    """Resolve a grouping once into a function that labels the pairs of any run.

    Use this instead of :func:`group_labels` to score several runs, so the dataset annotation is looked up once.

    :param grouping: As for :func:`group_labels`.
    :param dataset: As for :func:`group_labels`.
    :returns: A function from a run to its label array, as returned by :func:`group_labels`.
    :raises ValueError: As for :func:`group_labels`, before any run is labelled.
    """
    if grouping in _ID_ATTRIBUTE:

        def builtin(run: RunResult) -> np.ndarray:
            labels = np.asarray(getattr(run, _ID_ATTRIBUTE[grouping]))
            return _study_of(labels) if grouping == "study" else labels

        return builtin
    if dataset is None:
        raise ValueError(
            f"Unknown grouping {grouping!r} without a dataset: only {GROUPINGS} are built in. "
            "Pass dataset= to group by a sample or drug annotation column."
        )
    axis, column = _annotation_axis(grouping, dataset)
    annotation = _clean_annotation(dataset.sample_meta[column] if axis == "sample" else dataset.response.var[column])
    attribute = _ID_ATTRIBUTE[axis]

    def annotated(run: RunResult) -> np.ndarray:
        return _look_up(annotation, np.asarray(getattr(run, attribute)))

    return annotated


def _annotation_axis(grouping: str, dataset: Dataset) -> tuple[str, str]:
    """Find which axis of the dataset a non-built-in grouping names, and the column on it.

    :param grouping: ``"<column>"``, ``"sample:<column>"`` or ``"drug:<column>"``.
    :param dataset: Dataset whose annotation columns to search.
    :returns: ``("sample" | "drug", column)``.
    :raises ValueError: If the column is on neither axis, or a bare name is on both.
    """
    columns = {"sample": list(dataset.sample_meta.columns), "drug": list(dataset.response.var.columns)}
    prefix, separator, name = grouping.partition(":")
    if separator and prefix in _AXES:
        axes = [prefix] if name in columns[prefix] else []
    else:
        name = grouping
        axes = [axis for axis in _AXES if name in columns[axis]]
    if len(axes) > 1:
        raise ValueError(
            f"Grouping {grouping!r} is both a sample and a drug annotation; "
            f"use {f'sample:{name}'!r} or {f'drug:{name}'!r}."
        )
    if not axes:
        raise ValueError(
            f"Unknown grouping {grouping!r}: expected {GROUPINGS} or an annotation column. "
            f"Sample annotations: {columns['sample']}. Drug annotations: {columns['drug']}."
        )
    return axes[0], name


def _clean_annotation(column: pd.Series) -> pd.Series:
    """Turn an annotation column into string labels, NaN where there is no usable value.

    :param column: One ``obs`` or ``var`` column, indexed by ID.
    :returns: Object series indexed by the string ID (first row of a repeated ID); each value is the
        value as a string, or NaN where it is missing, empty or ``"unknown"`` (case-insensitive).
    """
    import pandas as pd

    column = column[~column.index.duplicated()]
    values = column.astype(object)
    text = values.astype(str)
    stripped = text.str.strip().str.lower()
    undefined = values.isna() | stripped.eq("") | stripped.eq(UNKNOWN_LABEL)
    labels = text.to_numpy(dtype=object)
    labels[undefined.to_numpy()] = np.nan
    return pd.Series(labels, index=column.index.astype(str), dtype=object)


def _look_up(annotation: pd.Series, ids: np.ndarray) -> np.ndarray:
    """Return the annotation of each ID, NaN for an ID that is not in ``annotation``.

    :param annotation: Cleaned annotation as returned by :func:`_clean_annotation`.
    :param ids: IDs of the pairs.
    :returns: Object array with one label or NaN per ID.
    """
    positions = annotation.index.get_indexer(ids.astype(str))
    values = np.append(annotation.to_numpy(dtype=object), np.nan)
    return values[positions]


def _study_of(sample_ids: np.ndarray) -> np.ndarray:
    """Return the prefix before ``"::"`` of every sample ID, NaN where there is none."""
    distinct, inverse = np.unique(sample_ids.astype(str), return_inverse=True)
    prefixes = np.empty(len(distinct), dtype=object)
    for position, sample_id in enumerate(distinct):
        study, separator, _ = sample_id.partition(STUDY_SEPARATOR)
        prefixes[position] = study if separator and study else np.nan
    return prefixes[inverse]


def defined_labels(labels: np.ndarray) -> np.ndarray:
    """Return which pairs have a group, i.e. a label that is not NaN.

    :param labels: Labels as returned by :func:`group_labels`.
    :returns: Boolean array, all True unless the labels are an object array holding NaN.
    """
    if labels.dtype != object:
        return np.ones(labels.shape, dtype=bool)
    return np.fromiter((isinstance(label, str) for label in labels), dtype=bool, count=len(labels))
