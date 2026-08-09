"""Runtime constants for dataset identifiers and measures."""

DRUG_IDENTIFIER = "pubchem_id"
CELL_LINE_IDENTIFIER = "cell_line_name"
TISSUE_IDENTIFIER = "tissue"
ALLOWED_MEASURES = ["LN_IC50", "EC50", "IC50", "pEC50", "AUC", "response"]
ALLOWED_MEASURES.extend([f"{m}_curvecurator" for m in ALLOWED_MEASURES])
