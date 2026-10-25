"""Runtime constants for dataset identifiers."""

DRUG_IDENTIFIER = "pubchem_id"
# Optional display column: only datasets whose samples are cell lines carry it, and nothing
# in drevalpy may require it (fall back to the obs index, i.e. the sample ID).
CELL_LINE_NAME_COLUMN = "cell_line_name"
TISSUE_IDENTIFIER = "tissue"
