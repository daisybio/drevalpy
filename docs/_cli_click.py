"""Docs-only helpers for generating the CLI reference from the Typer app.

Typer vendors its own Click classes, so ``sphinx-click`` cannot introspect the
app via ``isinstance(..., click.Command)``. Instead we render a nested RST
reference at Sphinx build time.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import click
from _generated_io import write_text_if_changed
from typer.main import get_command

from drevalpy.cli.main import app

DOCS_DIR = Path(__file__).resolve().parent
GENERATED_REFERENCE = DOCS_DIR / "cli" / "_generated_reference.rst"


def _format_default(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return " ".join(str(item) for item in value)
    text = str(value)
    if text in {"", "None"}:
        return None
    return text


def _param_body(param) -> str:
    """Phrase one parameter's definition-list body.

    :param param: Click parameter to describe.
    :returns: The help text, the rendered default, or a placeholder.
    """
    parts: list[str] = []
    help_text = (getattr(param, "help", None) or "").strip()
    if help_text:
        parts.append(help_text)
    default = _format_default(getattr(param, "default", None))
    if default is not None and not getattr(param, "required", False):
        parts.append(f"Default: ``{default}``.")
    return " ".join(parts) if parts else "No description."


def _documented_params(command) -> list:
    """Return the parameters worth documenting for *command*.

    Drops positional arguments, which carry no ``opts``, and the two
    shell-completion options Typer injects into every app.

    :param command: Click command to inspect.
    :returns: Click parameters in declaration order.
    """
    skip = {"install_completion", "show_completion"}
    return [
        param for param in command.params if getattr(param, "opts", None) and getattr(param, "name", None) not in skip
    ]


def _render_params(command) -> list[str]:
    """Render options as a definition list (no section titles).

    :param command: Click command whose parameters should be rendered
    :returns: RST lines for the parameter definition list
    """
    lines: list[str] = []
    for param in _documented_params(command):
        lines.append(" / ".join(f"``{opt}``" for opt in param.opts))
        lines.append(f"   {_param_body(param)}")
        lines.append("")
    return lines


def generate_cli_reference_rst() -> str:
    """Return RST for the full ``drevalpy`` CLI, including subcommands.

    :returns: generated CLI reference as an RST document string
    """
    root = cast(click.Group, get_command(app))
    lines = [
        "Root command",
        "------------",
        "",
        "Run the full experiment suite:",
        "",
        ".. code-block:: bash",
        "",
        "   drevalpy [OPTIONS]",
        "",
    ]
    help_text = (getattr(root, "help", None) or "").strip()
    if help_text:
        lines.extend([help_text, ""])
    lines.extend(_render_params(root))

    lines.extend(
        [
            "Subcommands",
            "-----------",
            "",
        ]
    )
    for name in sorted(root.commands):
        command = root.commands[name]
        heading = f"drevalpy {name}"
        lines.extend([heading, "~" * len(heading), ""])
        cmd_help = (getattr(command, "help", None) or "").strip()
        if cmd_help:
            lines.extend([cmd_help, ""])
        lines.extend(
            [
                ".. code-block:: bash",
                "",
                f"   drevalpy {name} [OPTIONS]",
                "",
            ]
        )
        lines.extend(_render_params(command))
    return "\n".join(lines).rstrip() + "\n"


def write_generated_cli_reference() -> Path:
    """Write the generated CLI reference RST consumed by ``cli/reference.rst``.

    :returns: path to the generated RST file
    """
    write_text_if_changed(GENERATED_REFERENCE, generate_cli_reference_rst())
    return GENERATED_REFERENCE


click_app = get_command(app)
