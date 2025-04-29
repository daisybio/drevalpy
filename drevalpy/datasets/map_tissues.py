"""
Command-line tool to generate and update harmonized tissue annotations for cancer cell lines.

Uses Cellosaurus and DepMap metadata (and some errors in these datasets are fixed manually)
and adds it to the response datasets e.g. for LTO splits.
Use it for reference or to create the tissue annotations for custom datasets

This tool performs the following steps:
1. Loads all unique Cellosaurus IDs from a specified set of drug response datasets (e.g., CCLE, GDSC1).
2. Downloads and parses the latest Cellosaurus reference file to extract metadata
(name, site of derivation, and disease).
3. Loads DepMap sample information to merge and normalize disease labels.
4. Applies a curated tissue synonym dictionary to map specific
diseases to broader, biologically meaningful tissue categories.
5. Manually overrides mappings for key misclassified or ambiguous cell lines
based on verified external sources (e.g., ATCC, NCI).
6. Saves the final tissue mapping to a central CSV file.
7. Propagates the harmonized tissue column back into each of the original datasets.

Arguments:
- `data_path` (str): Path to the directory containing all datasets and metadata files.
- `dataset` (str): One of {"CCLE", "GDSC1", "GDSC2", "CTRPv1", "CTRPv2", "all"}
- save_tissue_mapping (bool): If True, saves the tissue mapping to a CSV file.
or a custom dataset name. When "all" is specified, the tissue mapping is applied across all non-custom datasets

Notes:
- This script assumes each dataset is stored as a CSV file under `<data_path>/<dataset>/<dataset>.csv`.
- Tissue mapping is derived from a curated synonym dictionary and adjusted
for special cases based on literature or database references.
- The output `tissue_mapping.csv` is saved in `<data_path>/meta/`.

Example usage:
    python -m mymodule.add_tissue_mapping data/ all
"""

import argparse
import os
import urllib.request

import pandas as pd

from . import AVAILABLE_DATASETS
from .loader import download_dataset

_tissue_synonyms = {
    "Lung Non Small Cell": [
        "bronchogenic carcinoma",
        "lung",
        "lung adenocarcinoma",
        "lung carcinoma",
        "lung squamous cell carcinoma",
        "lung carcinoid tumor",
        "lung giant cell carcinoma",
        "lung adenosquamous carcinoma",
        "lung non-small cell carcinoma",
        "lung large cell carcinoma",
        "lung sarcomatoid carcinoma",
        "lung papillary adenocarcinoma",
        "mesothelioma",
        "minimally invasive lung adenocarcinoma",
    ],
    "Lung Small Cell": [
        "small cell lung cancer",
    ],
    "Lung Mesothelial": [
        "pleural mesothelial cells",
    ],
    "Lung Mediastinal": [
        "thoracic smarca4-deficient undifferentiated tumor",
    ],
    "Lung Salivary": [
        "cystic fibrosis",
        "lung mucoepidermoid carcinoma",
    ],
    "Colon": [
        "colorectal",
        "colon adenocarcinoma",
        "colon carcinoma",
        "cecum adenocarcinoma",
        "gardner syndrome",
        "intestine",
        "rectal adenocarcinoma",
    ],
    "Small Intestine": [
        "small_intestine",
        "small intestine adenocarcinoma",
        "small intestine carcinoid tumor",
        "small intestine neuroendocrine tumor",
        "small intestine neuroendocrine carcinoma",
        "duodenal adenocarcinoma",
        "somatostatinoma",
    ],
    "Stomach Non Small Cell": [
        "familial adenomatous polyposis",
        "gastric",
        "gastric adenocarcinoma",
        "gastric adenosquamous carcinoma",
        "gastric carcinoma",
        "gastric choriocarcinoma",
        "gastric fundus carcinoma",
        "gastric mucinous adenocarcinoma",
        "gastric signet ring cell adenocarcinoma",
        "gastric tubular adenocarcinoma",
        "klatskin tumor",
    ],
    "Stomach Small Cell": [
        "gastric small cell carcinoma",
    ],
    "Head and neck": [
        "adenocarcinoma of the esophagus",
        "carcinoma of esophagus",
        "esophagus",
        "head and neck squamous cell carcinoma",
        "oral epithelial dysplasia",
        "parotid gland mucoepidermoid carcinoma",
        "squamous cell carcinoma of the larynx",
        "squamous cell carcinoma of the hypopharynx",
        "squamous cell carcinoma of the oral cavity",
        "squamous cell carcinoma of the oral tongue",
        "squamous cell carcinoma of the salivary glands",
        "squamous cell carcinoma of salivary glands",
        "squamous cell carcinoma of the esophagus",
        "tongue adenosquamous carcinoma",
        "upper_aerodigestive",
    ],
    "Skin Melanoma": [
        "skin",
        "melanoma",
        "cutaneous melanoma",
        "amelanotic melanoma",
        "vulvar melanoma",
        "vaginal melanoma",
    ],
    "Skin": [
        "skin squamous cell carcinoma",
    ],
    "Blood Myeloid": [
        "acute erythroid leukemia",
        "acute megakaryoblastic leukemia",
        "acute megakaryoblastic leukemia in Down syndrome",
        "acute monoblastic/monocytic leukemia",
        "acute myelomonocytic leukemia",
        "acute myeloblastic leukemia with maturation",
        "acute myeloid leukemia",
        "acute promyelocytic leukemia",
        "bone marrow",
        "chronic eosinophilic leukemia",
        "chronic myeloid leukemia",
        "leukemia",
        "mixed phenotype acute leukemia",
        "myelodysplastic syndrome",
    ],
    "Blood Lymphoid": [
        "adult b acute lymphoblastic leukemia",
        "b-lymphoblastic leukemia/lymphoma with t(1",
        "b-lymphoblastic leukemia/lymphoma with t(12",
        "b-lymphoblastic leukemia/lymphoma with t(9",
        "b-lymphoblastic leukemia/lymphoma with t(v",
        "b-lymphoblastic leukemia/lymphoma with t(17",
        "b-cell prolymphocytic leukemia",
        "b-cell chronic lymphocytic leukemia",
        "b-cell non-hodgkin lymphoma",
        "burkitt lymphoma",
        "classic hairy cell leukemia",
        "diffuse large b-cell lymphoma",
        "follicular lymphoma",
        "hereditary spherocytosis",
        "hodgkin lymphoma",
        "non-hodgkin lymphoma",
        "mantle cell lymphoma",
        "natural killer cell lymphoblastic leukemia/lymphoma",
        "precursor b-cell acute lymphoblastic leukemia",
        "precursor t-cell acute lymphoblastic leukemia",
        "primary cutaneous t-cell lymphoma",
        "primary effusion lymphoma",
        "primary mediastinal large b-cell lymphoma",
        "t-cell large granular lymphocyte leukemia",  # TODO split up in B&T?
        "t-cell non-hodgkin lymphoma",
        "splenic marginal zone lymphoma",
        "sezary syndrome",
    ],
    "Blood Plasma": [
        "multiple myeloma",
    ],
    "Blood: Large Cell": [
        "alk-positive anaplastic large cell lymphoma",
    ],
    "Bone": ["bone", "osteosarcoma", "chondrosarcoma", "extraskeletal myxoid chondrosarcoma", "chordoma"],
    "Bone: Child": ["ewing sarcoma"],
    "Muscle": [
        "rhabdomyosarcoma",
        "rhabdoid",
        "rhabdoid tumor",
        "leiomyosarcoma",
        "atypical teratoid rhabdoid tumor",
        "alveolar rhabdomyosarcoma",
    ],
    "Soft Tissue": [
        "epithelioid sarcoma",  # TODO change?
        "soft_tissue",
        "fibrosarcoma",
        "undifferentiated pleomorphic sarcoma",
        "liposarcoma",
        "fibroblast",
        "synovial sarcoma",
    ],
    "Thyroid": [
        "thyroid",
        "anaplastic thyroid carcinoma",
        "differentiated thyroid carcinoma",
        "multiple endocrine neoplasia type 2",
        "adrenocortical carcinoma",
        "thyroid carcinoma",
    ],
    "Brain": [
        "central_nervous_system",
        "glioblastoma",
        "gliosarcoma",
        "astrocytoma",
        "anaplastic astrocytoma",
        "diffuse astrocytoma",
        "medulloblastoma",
        "oligodendroglioma",
    ],
    "Nervous system": [
        "neuroblastoma",
        "neurofibromatosis type 1",
        "primitive neuroectodermal tumor",
        "peripheral primitive neuroectodermal tumor",
        "peripheral_nervous_system",
        "nervous system",
    ],
    "Breast": [
        "breast",
        "breast adenocarcinoma",
        "breast acantholytic squamous cell carcinoma",
        "breast carcinoma",
        "breast ductal carcinoma",
        "invasive breast lobular carcinoma",
        "invasive breast carcinoma of no special type",
        "hereditary breast and ovarian cancer syndrome",
    ],
    "Ovary": [
        "ovary",
        "adenocarcinoma of ovary",
        "clear cell adenocarcinoma of the ovary",
        "high grade ovarian serous adenocarcinoma",
        "maligant granulosa cell tumor of the ovary",
        "malignant ovarian Brenner tumor",
        "mucinous adenocarcinoma of ovary",
        "ovarian carcinoma",
        "ovarian serous adenocarcinoma",
        "ovarian serous cystadenocarcinoma",
        "small cell carcinoma of the ovary",
    ],
    "Uterus": [
        "uterus",
        "high-grade neuroendocrine carcinoma of the cervix uteri",
        "carcinosarcoma of the corpus uteri",
        "leiomyosarcoma of the corpus uteri",
        "sarcoma of the corpus uteri",
    ],
    "Endometrial": [
        "endometrial adenocarcinoma",
        "endometrial adenosquamous carcinoma",
        "endometrial carcinoma",
        "endometrial clear cell adenocarcinoma",
        "endometrioid carcinoma of ovary",
        "endometrial endometrioid adenocarcinoma, variant with squamous differentiation",
        "endometrial stromal sarcoma",
        "type i endometrial adenocarcinoma",
        "type ii endometrial adenocarcinoma",
    ],
    "Cervix": [
        "cervix",
        "cervical carcinoma",
        "human papillomavirus-related endocervical adenocarcinoma",
        "vulvar carcinoma",  # TODO this is not cervix
        "vulvar squamous cell carcinoma",
        "squamous cell carcinoma of the cervix uteri",
    ],
    "Prostate": ["prostate carcinoma", "benign prostatic hyperplasia"],
    "Prostate Small Cell": ["prostate small cell carcinoma"],
    "Kidney": [
        "kidney",
        "kidney neoplasm",
        "renal cell carcinoma",
        "clear cell renal carcinoma",
        "papillary renal cell carcinoma",
    ],
    "Bladder": ["urinary_tract", "bladder carcinoma", "bladder squamous cell carcinoma", "recurrent bladder carcinoma"],
    "Liver": [
        "adult hepatocellular carcinoma",
        "liver",
        "liver and intrahepatic bile duct epithelial neoplasm",
        "cholangiocarcinoma",
        "bile_duct",
        "hepatoblastoma",
        "carcinoma of gallbladder and extrahepatic biliary tract",
        "pediatric hepatocellular carcinoma",
    ],
    "Pancreas": [
        "carcinoma of the ampulla of vater",
        "pancreas",
        "pancreatic adenocarcinoma",
        "pancreatic carcinoma",
        "pancreatic ductal adenocarcinoma",
        "pancreatic undifferentiated carcinoma",
        "squamous cell carcinoma of pancreas",
    ],
    "Adrenal Gland": ["adrenal_cortex"],
    "Embryonic": [
        "embryo",
        "non-central nervous system-localized embryonal carcinoma",
        "embryonal carcinoma",
        "embryonal rhabdomyosarcoma",
        "gestational choriocarcinoma",
        "mixed germ cell tumor",
    ],
    "Urothelium": [
        "upper tract urothelial carcinoma",
    ],
    "Unknown": ["unknown", "other", "carcinoid syndrome", "down syndrome"],
}


def _parse_cellosaurus(cellosaurus_path: str) -> tuple[dict, dict, dict]:
    """
    Parse Cellosaurus file and return mappings from cellosaurus ID to name, site, and disease.

    :param cellosaurus_path: Path to the Cellosaurus text file
    :return: Tuple of dictionaries (id_to_name, id_to_site, id_to_disease)
    """
    id_to_name, id_to_site, id_to_disease = {}, {}, {}

    with open(cellosaurus_path, encoding="utf-8") as f:
        current_ids, current_name, site, disease = [], None, None, None
        for line in f:
            if line.startswith("ID   "):
                current_name = line.strip().split("   ")[1]
            elif line.startswith("AC   "):
                current_ids = [s.strip() for s in line[5:].split(";") if s.strip()]
            elif line.startswith("CC   Derived from site:"):
                parts = line.strip().split(":", 1)[1].split(";")
                if len(parts) >= 2:
                    site = parts[1].strip()
            elif line.startswith("DI   ") and current_ids:
                parts = line[5:].split(";")
                if len(parts) >= 3:
                    disease = parts[2].strip()
            elif line.strip() == "//":
                for cid in current_ids:
                    if current_name:
                        id_to_name[cid] = current_name
                    if site:
                        id_to_site[cid] = site
                    if disease:
                        id_to_disease[cid] = disease
                current_ids, current_name, site, disease = [], None, None, None

    return id_to_name, id_to_site, id_to_disease


def _apply_manual_cell_line_corrections(tissue_map: pd.Series) -> pd.Series:
    """Apply manual tissue corrections for misclassified or ambiguous cell lines with documented sources.

    :param tissue_map: Series mapping Cellosaurus IDs to tissues
    :return: Updated tissue mapping Series
    """
    manual_entries = [
        # CVCL_0977: Hs 888.Lu
        # Source: Cell Model Passports - SIDM01745
        # https://cellmodelpassports.sanger.ac.uk/passports/SIDM01745
        ("CVCL_0977", "Lung"),
        # CVCL_1072: ARH-77
        # Source: Culture Collections - ARH-77
        # https://www.culturecollections.org.uk/products/celllines/detail.jsp?refId=88121201
        ("CVCL_1072", "Blood"),
        # CVCL_1305: IM-9
        # Source: ATCC - CCL-159
        # https://www.atcc.org/products/ccl-159
        ("CVCL_1305", "Blood"),
        # CVCL_1665: RPMI-6666
        # Source: https://scicrunch.org/resolver/RRID:CVCL_1665?q=&i=rrid:cvcl_1665-kclb-10113
        # Derived from leukemic cells of a myeloma patient, EBV-transformed B lymphoblastoid line
        ("CVCL_1665", "Blood"),
        # CVCL_0807: Hs 578Bst
        # Source: https://scicrunch.org/resolver/CVCL_0807/
        ("CVCL_0807", "Breast"),
        # CVCL_L296: H-STS
        # Source: https://www.nature.com/articles/s41588-019-0490-z
        ("CVCL_L296", "Blood"),
        # CVCL_ZA06: WT2-iPS
        # Source: https://discover.nci.nih.gov/rsconnect/cellminercdb/cell_lines/wt2ips_cellminercdb.html
        # NOTE: Excluded on purpose — cl doesn't exist, was wrong mapping before
        # ("CVCL_ZA06", "Skin"),
        # CVCL_L298: P-STS
        # Source: Pfragner R, Behmel A, Höger H, Beham A,
        # Ingolic E, Stelzer I, Svejda B, Moser VA, Obenauf AC, Siegl V, et al. (2009).
        # Establishment and characterization of three novel cell lines
        # – P-STS, L-STS, H-STS – derived from a human metastatic midgut carcinoid.
        # Anticancer Research, 29(6), 1951–1961.
        ("CVCL_L298", "Small Intestine"),
        # CVCL_3386 was misclassified.
        # source: https://www.cellosaurus.org/CVCL_3386
        ("CVCL_3386", "Blood"),
    ]

    for cellosaurus_id, tissue in manual_entries:
        tissue_map[cellosaurus_id] = tissue

    return tissue_map


def _harmonize_disease_annotations(df_cellosaurus: pd.DataFrame, sample_info: pd.DataFrame) -> pd.DataFrame:
    """Merge Cellosaurus and DepMap data, normalize names, and harmonize disease annotations.

    :param df_cellosaurus: DataFrame containing Cellosaurus data
    :param sample_info: DataFrame containing DepMap sample information
    :return: Merged DataFrame with harmonized disease annotations

    """
    df_cellosaurus["name_norm"] = df_cellosaurus["cell_line_name"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    sample_info["name_norm"] = (
        sample_info["stripped_cell_line_name"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    )

    merged = pd.merge(df_cellosaurus, sample_info, on="name_norm", how="left")
    merged["disease"] = merged["disease"].replace(r"^\s*$", "unknown", regex=True)

    merged["disease_combined"] = (
        merged.apply(
            lambda row: (
                row["cellosaurus_disease"]
                if (pd.isna(row["disease"]) or row["disease"].strip().lower() == "unknown")
                and pd.notna(row["cellosaurus_disease"])
                else row["disease"]
            ),
            axis=1,
        )
        .str.strip()
        .str.lower()
    )

    return merged


def main():
    """Main function to add tissue mapping to datasets."""
    parser = argparse.ArgumentParser(description="Add tissue mapping to datasets")
    parser.add_argument("data_path", help="Path to dataset root directory", default="data")
    parser.add_argument("dataset", help="Dataset name (e.g., CCLE) or 'all'", default="all")
    parser.add_argument(
        "--save_tissue_mapping",
        action="store_true",
        help="Save the tissue mapping to a CSV file",
    )
    args = parser.parse_args()
    data_path = args.data_path
    dataset = args.dataset
    save_tissue_mapping = args.save_tissue_mapping
    if dataset != "all":
        datasets = [dataset]
    else:
        datasets = AVAILABLE_DATASETS.keys()

    cell_lines = []

    # Load all unique Cellosaurus IDs from available datasets
    for ds in datasets:
        csv_path = os.path.join(data_path, ds, f"{ds}.csv")
        try:
            df = pd.read_csv(csv_path, dtype=str, low_memory=False)
            cell_lines.extend(df["cellosaurus_id"].dropna().unique())
        except FileNotFoundError:
            continue

    cellosaurus_ids = pd.Series(cell_lines).drop_duplicates().reset_index(drop=True)

    #  Download Cellosaurus if needed
    cellosaurus_path = os.path.join(data_path, "meta", "cellosaurus.txt")
    cellosaurus_path.parent.mkdir(parents=True, exist_ok=True)

    if not cellosaurus_path.exists():
        url = "https://ftp.expasy.org/databases/cellosaurus/cellosaurus.txt"
        urllib.request.urlretrieve(url, cellosaurus_path)  # noqa-S310

    # Parse Cellosaurus
    id_to_name, id_to_site, id_to_disease = _parse_cellosaurus(cellosaurus_path)

    # Build Cellosaurus DataFrame
    df_cellosaurus = pd.DataFrame(
        {
            "cellosaurus_id": cellosaurus_ids,
            "cell_line_name": cellosaurus_ids.map(id_to_name),
            "cellosaurus_derived_from_site": cellosaurus_ids.map(id_to_site),
            "cellosaurus_disease": cellosaurus_ids.map(id_to_disease),
        }
    ).dropna(subset=["cell_line_name"])

    # Load DepMap sample_info
    depmap_path = os.path.join(data_path, "meta", "DepMap_sample_info.csv")
    if not os.path.exists(depmap_path):
        download_dataset(dataset_name="meta", data_path=data_path, redownload=True)

    sample_info = pd.read_csv(depmap_path, dtype=str, low_memory=False)

    merged = _harmonize_disease_annotations(df_cellosaurus, sample_info)

    # Synonym mapping

    tissue_lookup = {syn.lower(): tissue for tissue, syns in _tissue_synonyms.items() for syn in syns}

    # Map tissues
    merged["disease_cleaned"] = merged["disease_combined"].map(tissue_lookup).fillna("Unknown").str.title()

    # Final tissue map
    final = merged[
        [
            "cellosaurus_id",
            "cell_line_name",
            "DepMap_ID",
            "disease",
            "disease_combined",
            "disease_cleaned",
            "disease_sutype",
            "disease_sub_subtype",
            "culture_type",
            "culture_medium",
            "gender",
            "source",
            "cellosaurus_derived_from_site",
            "cellosaurus_disease",
        ]
    ]

    # Make sure the mapping has unique index entries
    tissue_map = final.drop_duplicates(subset="cellosaurus_id").set_index("cellosaurus_id")["disease_cleaned"]

    tissue_map = _apply_manual_cell_line_corrections(tissue_map)

    final.loc[:, "tissue"] = final.loc[:, "cellosaurus_id"].map(tissue_map)
    final = final.copy()
    if save_tissue_mapping:
        final.drop_duplicates(subset="cellosaurus_id", inplace=True)
        tissue_mapping_path = os.path.join(data_path, "meta", "tissue_mapping.csv")
        final.to_csv(tissue_mapping_path, index=False)

    # Add tissue column to each dataset
    for ds in datasets:
        path = os.path.join(data_path, ds, f"{ds}.csv")
        if not os.path.exists(path):
            print(f"Dataset {path} not found, skipping.")
            continue

        df = pd.read_csv(path, low_memory=False)

        df["tissue"] = df["cellosaurus_id"].map(tissue_map)

        df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
