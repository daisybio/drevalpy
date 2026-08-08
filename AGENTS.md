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
