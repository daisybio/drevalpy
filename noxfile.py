"""Nox sessions."""

import os
import shlex
import shutil
import sys
from pathlib import Path
from textwrap import dedent

import nox
from rich import print

try:
    from nox_poetry import Session, session
except ImportError:
    print("[bold red]Did not find nox-poetry installed in your current environment!")
    print("[bold blue]Try installing it using [bold green]pip install nox-poetry [bold blue]! ")
    sys.exit(1)

package = "drevalpy"
python_versions = ["3.12", "3.13"]
nox.options.sessions = (
    "pre-commit",
    "mypy",
    "tests",
    "xdoctest",
    "docs-build",
)


def activate_virtualenv_in_precommit_hooks(session: Session) -> None:
    """Activate virtualenv in hooks installed by pre-commit.

    This function patches git hooks installed by pre-commit to activate the
    session's virtual environment. This allows pre-commit to locate hooks in
    that environment when invoked from git.

    :param session: The Session object.
    """
    assert session.bin is not None  # noqa: S101

    # Only patch hooks containing a reference to this session's bindir. Support
    # quoting rules for Python and bash, but strip the outermost quotes so we
    # can detect paths within the bindir, like <bindir>/python.
    bindirs = [
        bindir[1:-1] if bindir[0] in "'\"" else bindir for bindir in (repr(session.bin), shlex.quote(session.bin))
    ]

    virtualenv = session.env.get("VIRTUAL_ENV")
    if virtualenv is None:
        return

    headers = {
        # pre-commit < 2.16.0
        "python": f"""\
            import os
            os.environ["VIRTUAL_ENV"] = {virtualenv!r}
            os.environ["PATH"] = os.pathsep.join((
                {session.bin!r},
                os.environ.get("PATH", ""),
            ))
            """,
        # pre-commit >= 2.16.0
        "bash": f"""\
            VIRTUAL_ENV={shlex.quote(virtualenv)}
            PATH={shlex.quote(session.bin)}"{os.pathsep}$PATH"
            """,
        # pre-commit >= 2.17.0 on Windows forces sh shebang
        "/bin/sh": f"""\
            VIRTUAL_ENV={shlex.quote(virtualenv)}
            PATH={shlex.quote(session.bin)}"{os.pathsep}$PATH"
            """,
    }

    hookdir = Path(".git") / "hooks"
    if not hookdir.is_dir():
        return

    for hook in hookdir.iterdir():
        if hook.name.endswith(".sample") or not hook.is_file():
            continue

        if not hook.read_bytes().startswith(b"#!"):
            continue

        text = hook.read_text()

        if not any(Path("A") == Path("a") and bindir.lower() in text.lower() or bindir in text for bindir in bindirs):
            continue

        lines = text.splitlines()

        for executable, header in headers.items():
            if executable in lines[0].lower():
                lines.insert(1, dedent(header))
                hook.write_text("\n".join(lines))
                break


@session(name="pre-commit", python=python_versions)
def precommit(session: Session) -> None:
    """
    Lint using pre-commit.

    :param session: The Session object.
    """
    args = session.posargs or ["run", "--all-files"]
    session.install(
        "black",
        "flake8",
        "flake8-bandit",
        "flake8-bugbear",
        "flake8-docstrings",
        "darglint",
        "flake8-rst-docstrings",
        "isort",
        "pep8-naming",
        "pre-commit",
        "pre-commit-hooks",
        "pyupgrade",
    )
    session.run("pre-commit", *args)
    if args and args[0] == "install":
        activate_virtualenv_in_precommit_hooks(session)


@session(python=python_versions)
def mypy(session: Session) -> None:
    """
    Type-check using mypy.

    :param session: The Session object.
    """
    args = session.posargs or ["drevalpy", "tests", "docs/conf.py"]
    session.install(".")
    session.install("mypy", "pytest", "types-requests", "types-attrs", "types-PyYAML", "types-toml")
    session.run("mypy", *args)


@session(python=python_versions)
def tests(session: Session) -> None:
    """
    Run the test suite.

    :param session: The Session object.
    """
    session.install(".[fit]")
    session.install("coverage[toml]", "pytest", "pygments")
    try:
        session.run("coverage", "run", "--parallel", "-m", "pytest", *session.posargs)
    finally:
        if session.interactive:
            session.notify("coverage")


@session
def coverage(session: Session) -> None:
    """
    Produce the coverage report.

    :param session: The Session object.
    """
    # Do not use session.posargs unless this is the only session.
    nsessions = len(session._runner.manifest)  # type: ignore[attr-defined]
    has_args = session.posargs and nsessions == 1
    args = session.posargs if has_args else ["report", "-i"]

    session.install("coverage[toml]")

    if not has_args and any(Path().glob(".coverage.*")):
        session.run("coverage", "combine")

    session.run("coverage", *args)


@session(python=python_versions)
def typeguard(session: Session) -> None:
    """
    Runtime type checking using Typeguard.

    :param session: The Session object.
    """
    session.install(".[fit]")
    session.install("pytest", "typeguard", "pygments")
    session.run("pytest", f"--typeguard-packages={package}", *session.posargs)


@session(python=python_versions)
def xdoctest(session: Session) -> None:
    """
    Run examples with xdoctest.

    :param session: The Session object.
    """
    args = session.posargs or ["all"]
    session.install(".")
    session.install("xdoctest[colors]")
    session.run("python", "-m", "xdoctest", package, *args)


@session(name="docs-build", python=python_versions)
def docs_build(session: Session) -> None:
    """
    Build the documentation.

    :param session: The Session object.
    """
    args = session.posargs or ["docs", "docs/_build"]
    session.install("-r", "./docs/requirements.txt")

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-build", *args)


@session(python=python_versions)
def docs(session: Session) -> None:
    """
    Build and serve the documentation with live reloading on file changes.

    :param session: The Session object.
    """
    args = session.posargs or ["--open-browser", "docs", "docs/_build"]
    session.install(".")
    session.install(
        "sphinx",
        "sphinx-autobuild",
        "sphinx-click",
        "sphinx-rtd-theme",
        "sphinx-rtd-dark-mode",
    )

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-autobuild", *args)
