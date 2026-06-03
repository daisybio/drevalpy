"""Argv translation for the legacy ``drevalpy-fit-curves`` console script."""

from __future__ import annotations

from pathlib import Path

_VALUE_FLAG_ALIASES = {
    "--output-dir": "--output-dir",
    "--output_dir": "--output-dir",
    "--dataset-name": "--dataset-name",
    "--dataset_name": "--dataset-name",
    "--cores": "--cores",
    "--device": "--device",
    "--chunk-size": "--chunk-size",
    "--chunk_size": "--chunk-size",
    "--gpu-min-curves": "--gpu-min-curves",
    "--gpu_min_curves": "--gpu-min-curves",
    "--gpu-chunk-size": "--gpu-chunk-size",
    "--gpu_chunk_size": "--gpu-chunk-size",
}
_VALUE_FLAGS = tuple(dict.fromkeys(_VALUE_FLAG_ALIASES.values()))


def dataset_name_from_input(input_file: Path) -> str:
    """Infer dataset name from a raw viability CSV path."""
    return input_file.stem.removesuffix("_raw")


def forward_fit_curves_argv(argv: list[str]) -> list[str]:
    """Map ``drevalpy-fit-curves`` argv to ``drevalpy curation`` argv.

    :param argv: Tokens after the program name.
    :returns: Tokens for ``drevalpy curation``.
    :raises ValueError: If argv contains unknown or duplicate positional arguments.
    """
    input_file: str | None = None
    options: dict[str, str] = {}
    boolean_flags: list[str] = []
    index = 0

    while index < len(argv):
        token = argv[index]
        if token == "--normalize":
            boolean_flags.append(token)
            index += 1
            continue
        if token == "--gpu-available":
            boolean_flags.append(token)
            index += 1
            continue
        if token == "--no-gpu-available":
            boolean_flags.append(token)
            index += 1
            continue
        if token in _VALUE_FLAG_ALIASES:
            options[_VALUE_FLAG_ALIASES[token]] = argv[index + 1]
            index += 2
            continue
        if token.startswith("-"):
            raise ValueError(f"Unknown option for drevalpy-fit-curves: {token}")
        if input_file is None:
            input_file = token
            index += 1
            continue
        raise ValueError(f"Unexpected extra argument for drevalpy-fit-curves: {token}")

    if input_file is None:
        return ["curation", *boolean_flags]

    resolved_input = Path(input_file).expanduser().resolve()
    if "--output-dir" not in options:
        options["--output-dir"] = str(resolved_input.parent)
    if "--dataset-name" not in options:
        options["--dataset-name"] = dataset_name_from_input(resolved_input)

    forwarded = ["curation", "--input-file", str(resolved_input)]
    for flag in _VALUE_FLAGS:
        if flag in options:
            forwarded.extend([flag, options[flag]])
    forwarded.extend(boolean_flags)
    return forwarded
