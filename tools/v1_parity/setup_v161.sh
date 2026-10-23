#!/usr/bin/env bash
# Create the 1.6.1 reference checkout and its isolated CPU-only virtualenv.
#
# Usage: tools/v1_parity/setup_v161.sh
# Environment overrides:
#   V161_DIR      worktree location          (default /tmp/opencode/v161)
#   V161_COMMIT   pinned 1.x commit          (default f5ff3c70)
#   V161_PYTHON   interpreter 1.6.1 supports (default 3.12; 1.6.1 needs >=3.11,<3.14)
#   V161_EXTRAS   optional 1.6.1 extras      (default: rdkit xgboost lightgbm gseapy mygene obonet)
# The script only ever adds a worktree for V161_DIR; it never checks out 1.6.1 in the main tree.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
V161_DIR="${V161_DIR:-/tmp/opencode/v161}"
V161_COMMIT="${V161_COMMIT:-f5ff3c70}"
V161_PYTHON="${V161_PYTHON:-3.12}"
V161_EXTRAS="${V161_EXTRAS-rdkit xgboost lightgbm gseapy mygene obonet}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

if [ -e "$V161_DIR/.git" ]; then
    echo "reusing worktree $V161_DIR"
else
    mkdir -p "$(dirname "$V161_DIR")"
    git -C "$REPO_ROOT" worktree add --detach "$V161_DIR" "$V161_COMMIT"
fi
head="$(git -C "$V161_DIR" rev-parse HEAD)"
case "$head" in
    "$(git -C "$REPO_ROOT" rev-parse "$V161_COMMIT")") ;;
    *) echo "worktree is at $head, expected $V161_COMMIT" >&2; exit 1 ;;
esac

VENV="$V161_DIR/.venv"
# uv reads pyproject.toml/uv.toml from the working directory: stay out of the v2 tree (its
# `exclude-newer` cutoff would hide the versions 1.6.1 pinned).
cd "$V161_DIR"
[ -x "$VENV/bin/python" ] || uv venv --python "$V161_PYTHON" "$VENV"

# requirements.txt is a `poetry export` with the CUDA wheels pinned. Drop those and
# torch/torch-geometric lines, then install CPU torch separately at the pinned version.
torch_version="$(grep -E '^torch==' "$V161_DIR/requirements.txt" | head -1 | sed -E 's/^torch==([^ ;]+).*/\1/')"
filtered="$VENV/requirements.cpu.txt"
grep -v -E '^(nvidia-|cuda-|triton|torch==|poetry|build==|dulwich|keyring|jeepney|secretstorage)' \
    "$V161_DIR/requirements.txt" > "$filtered"

uv pip install --python "$VENV/bin/python" --index-url "$TORCH_INDEX" "torch==${torch_version}+cpu" \
    || uv pip install --python "$VENV/bin/python" --index-url "$TORCH_INDEX" torch
uv pip install --python "$VENV/bin/python" -r "$filtered"
if [ -n "$V161_EXTRAS" ]; then
    # shellcheck disable=SC2086
    uv pip install --python "$VENV/bin/python" $V161_EXTRAS
fi
uv pip install --python "$VENV/bin/python" -e "$V161_DIR" --no-deps

"$VENV/bin/python" - <<PY
import importlib.metadata as md
import sys

import numpy
import sklearn
import torch

import drevalpy

print("python", sys.version.split()[0])
print("torch", torch.__version__, "numpy", numpy.__version__, "sklearn", sklearn.__version__)
print("drevalpy", md.version("drevalpy"), "from", drevalpy.__file__)
PY
echo "1.6.1 reference ready: $VENV/bin/python (commit $head)"
