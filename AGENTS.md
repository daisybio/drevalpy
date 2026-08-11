# Agent notes

## Running prek / committing

Hooks are managed by prek (a fast, drop-in replacement for pre-commit). Run all hooks:

```bash
uv run prek run --all-files
```

Install the git hook so it runs automatically on commit:

```bash
uv run prek install
```

## Running tests

```bash
uv run pytest
```

## Building docs

```bash
uv run --group docs sphinx-build -W docs docs/_build
```

## Path handling

Use `universal_pathlib` (`UPath`) instead of `pathlib.Path` throughout the codebase. This enables transparent access to remote filesystems (S3, GCS, etc.). Import as:

```python
from upath import UPath
```

Note: Typer does not support `UPath` in CLI parameter annotations. Use `str` for path parameters and convert to `UPath` inside the function body.
