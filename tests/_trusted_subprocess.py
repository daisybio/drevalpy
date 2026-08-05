"""Trusted subprocess boundary for test isolation checks."""

from __future__ import annotations

import subprocess  # noqa: S404
import sys
from collections.abc import Sequence
from typing import Any


def run_trusted_python(
    script: str,
    *,
    cwd: str | None = None,
    extra_args: Sequence[str] | None = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    """Run an inline Python script in a fresh interpreter for test isolation.

    :param script: Python source executed via ``python -c``.
    :param cwd: Optional working directory for the child process.
    :param extra_args: Additional argv entries appended after ``-c`` and *script*.
    :param kwargs: Forwarded to ``subprocess.run`` (except ``check``).
    :returns: Completed process with captured stdout/stderr.
    """
    command = [sys.executable, "-c", script]
    if extra_args:
        command.extend(extra_args)
    return subprocess.run(  # noqa: S603
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=cwd,
        **kwargs,
    )
