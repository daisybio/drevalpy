"""Migrate existing .h5mu files to the new featurizer storage key conventions.

Reads an .h5mu file, renames old keys to match the featurizer's canonical
``storage_key``, creates the ``featurizer_variants`` uns registry, and writes
the updated file.
"""

from __future__ import annotations

import mudata as md
from upath import UPath

from drevalpy.log import get_logger

logger = get_logger(__name__)

_CELL_LINE_MODALITY_RENAMES: dict[str, str] = {
    # These are expected to already be canonical, listed for completeness
    "gene_expression": "gene_expression",
    "proteomics": "proteomics",
    "methylation": "methylation",
    "mutations": "mutations",
    "copy_number_variation": "copy_number_variation",
}

_DRUG_VARM_RENAMES: dict[str, str] = {
    "fingerprints": "morgan_fingerprint",
}

_UNS_RENAMES: dict[str, str] = {
    # Drug graphs and sparsego already canonical
    "drug_graphs": "drug_graphs",
}


def _rename_modalities(mdata: md.MuData) -> list[str]:
    """Rename cell-line modality keys if they use old naming."""
    changes: list[str] = []
    for old, new in _CELL_LINE_MODALITY_RENAMES.items():
        if old != new and old in mdata.mod and new not in mdata.mod:
            mdata.mod[new] = mdata.mod.pop(old)
            changes.append(f"modality: {old} -> {new}")
    return changes


def _rename_varm_keys(mdata: md.MuData) -> list[str]:
    """Rename drug varm keys in the response modality."""
    changes: list[str] = []
    if "response" not in mdata.mod:
        return changes
    response = mdata.mod["response"]
    if response.varm is None:
        return changes
    for old, new in _DRUG_VARM_RENAMES.items():
        if old != new and old in response.varm and new not in response.varm:
            response.varm[new] = response.varm[old]
            del response.varm[old]
            changes.append(f"varm: {old} -> {new}")
    return changes


def _rename_uns_keys(mdata: md.MuData) -> list[str]:
    """Rename uns keys if they use old naming."""
    changes: list[str] = []
    for old, new in _UNS_RENAMES.items():
        if old != new and old in mdata.uns and new not in mdata.uns:
            mdata.uns[new] = mdata.uns.pop(old)
            changes.append(f"uns: {old} -> {new}")
    return changes


def _build_featurizer_variants(mdata: md.MuData) -> dict[str, str]:
    """Build the featurizer_variants uns registry from present data."""
    variants: dict[str, str] = {}

    modality_keys = set(mdata.mod.keys()) - {"response"}
    for mod_key in sorted(modality_keys):
        variants[mod_key] = "modality"

    if "response" in mdata.mod:
        response = mdata.mod["response"]
        if response.varm is not None:
            for varm_key in sorted(response.varm.keys()):
                variants[varm_key] = "varm"

    for uns_key in sorted(mdata.uns.keys()):
        if uns_key == "featurizer_variants":
            continue
        if uns_key in ("drug_graphs", "sparsego"):
            variants[uns_key] = "uns"

    return variants


def align_mudata(input_path: str, output_path: str | None = None) -> None:
    """Align an existing .h5mu file to the new featurizer storage conventions.

    Args:
        input_path: Path to the existing .h5mu file.
        output_path: Path for the output file. If None, overwrites the input.
    """
    resolved_input = UPath(input_path)
    resolved_output = UPath(output_path) if output_path else resolved_input

    logger.info("Reading %s", resolved_input)
    md.set_options(pull_on_update=False)
    mdata = md.read_h5mu(resolved_input)

    all_changes: list[str] = []
    all_changes.extend(_rename_modalities(mdata))
    all_changes.extend(_rename_varm_keys(mdata))
    all_changes.extend(_rename_uns_keys(mdata))

    variants = _build_featurizer_variants(mdata)
    mdata.uns["featurizer_variants"] = variants
    all_changes.append(f"featurizer_variants registry: {len(variants)} entries")

    if all_changes:
        for change in all_changes:
            logger.info("  %s", change)
    else:
        logger.info("No renames needed; file already uses canonical keys.")

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    mdata.write(str(resolved_output))
    logger.info("Wrote aligned file to %s", resolved_output)
