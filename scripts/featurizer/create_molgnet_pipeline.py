"""Orchestration helper for the MolGNet embedding pipeline.

This script is a convenience wrapper that delegates to create_molgnet_embeddings.py.
For direct MuData usage, prefer running create_molgnet_embeddings.py directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from create_molgnet_embeddings import main as molgnet_main


def main() -> None:
    """CLI entry point for MolGNet pipeline."""
    parser = argparse.ArgumentParser(description="Run MolGNet embedding pipeline on a .h5mu file.")
    parser.add_argument("h5mu_path", type=Path, help="Path to the .h5mu file")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to MolGNet checkpoint (.pt)")
    parser.add_argument("--device", default=None, help="Torch device (e.g. cpu, cuda:0)")
    args = parser.parse_args()
    molgnet_main(args.h5mu_path, checkpoint=args.checkpoint, device=args.device)


if __name__ == "__main__":
    main()
