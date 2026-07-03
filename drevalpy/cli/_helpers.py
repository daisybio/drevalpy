"""Small helpers shared by Typer command modules."""

from __future__ import annotations

from argparse import Namespace
from typing import Any

ROOT_LIST_OPTIONS = frozenset(
    {
        "--models",
        "--baselines",
        "--test_mode",
        "--randomization_mode",
        "--cross_study_datasets",
    }
)

SUBCOMMAND_LIST_OPTIONS: dict[str, frozenset[str]] = {
    "evaluate-hpams": frozenset({"--hpam_yamls", "--pred_datas"}),
    "collect-results": frozenset({"--outfiles"}),
    "make-pipeline-report": frozenset({"--test_modes"}),
    "test-cv": frozenset({"--cross_study_datasets"}),
    "consolidate-single-drug": frozenset({"--cross_study_datasets"}),
}

KNOWN_SUBCOMMANDS = frozenset(SUBCOMMAND_LIST_OPTIONS) | frozenset(
    {
        "viability-preprocess",
        "viability-postprocess",
        "load-response",
        "make-cv-pkls",
        "make-hpam-yamls",
        "train-cv",
        "make-randomization-yamls",
        "make-final-split-pkls",
        "tune-final-model",
        "train-final-model",
        "evaluate-test",
        "report",
    }
)

# Backward-compatible alias used in tests and callers that override list options explicitly.
LIST_OPTIONS = ROOT_LIST_OPTIONS | frozenset().union(*SUBCOMMAND_LIST_OPTIONS.values())


def _is_option_token(token: str) -> bool:
    """Return whether *token* begins a new CLI option."""
    return token.startswith("-") and token not in {"-", "--"}


def _active_list_options(argv: list[str]) -> frozenset[str]:
    """Return list-option names valid for the command invoked by *argv*."""
    for token in argv:
        if not _is_option_token(token) and token in KNOWN_SUBCOMMANDS:
            return SUBCOMMAND_LIST_OPTIONS.get(token, frozenset())
    return ROOT_LIST_OPTIONS


def normalize_list_argv(argv: list[str], list_options: frozenset[str] | None = None) -> list[str]:
    """Expand argparse-style space-separated list options for Typer/Click.

    Converts ``--models A B`` into ``--models A --models B`` while leaving
    repeated-flag syntax unchanged. List-option names depend on whether the
    root pipeline or a subcommand is being invoked, so scalar flags such as
    ``--test_mode`` on ``test-cv`` are not expanded.

    :param argv: Command-line tokens without the program name.
    :param list_options: Optional override for tests; defaults to context-aware options.
    :return: Normalized argv suitable for Typer/Click list options.
    """
    active_options = list_options if list_options is not None else _active_list_options(argv)
    normalized: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token not in active_options:
            normalized.append(token)
            index += 1
            continue

        index += 1
        values: list[str] = []
        while index < len(argv) and not _is_option_token(argv[index]):
            values.append(argv[index])
            index += 1

        if not values:
            normalized.append(token)
            continue

        for value in values:
            normalized.extend([token, value])

    return normalized


def as_list(value: list[str] | tuple[str, ...] | None) -> list[str]:
    """Normalize Typer list options to a plain list.

    :param value: Optional sequence from a Typer multi-value option.
    :return: A plain list (empty when *value* is ``None``).
    """
    if value is None:
        return []
    return list(value)


def pipeline_namespace(**kwargs: Any) -> Namespace:
    """Build an ``argparse.Namespace`` for the full-pipeline entry point.

    :param kwargs: Pipeline option names and values.
    :return: Namespace consumed by ``drevalpy.utils.main``.
    """
    return Namespace(**kwargs)
