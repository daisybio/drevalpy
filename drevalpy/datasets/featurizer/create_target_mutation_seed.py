#!/usr/bin/env python3
"""Create target_mutation_seed.csv: drug -> biomarker gene -> sensitivity state, derived from databases.

The script takes **no input table**. The drugs are read directly from the datasets that drevalpy
provides (``drug_names.csv`` of every dataset in ``drevalpy.datasets.loader.AVAILABLE_DATASETS`` except
the derived ``*_clean*`` variants). A dataset's file is read from ``<data_path>/<dataset>/drug_names.csv``
if present; otherwise only that one file is fetched from the dataset's archive on Zenodo (HTTP range
requests, no full download), as drevalpy itself locates its datasets.

Everything else is derived at run time:

  * **ChEMBL** (ebi.ac.uk) -- the drug's molecular target genes: mechanism of action; if ChEMBL records
    none, the targets with potent binding-assay activity (pChEMBL >= 7, assay type B, human).
    The ChEMBL molecule is found via the PubChem CID (-> InChIKey) or, for drugs without a CID, by
    exact name.
  * **PubChem** (pubchem.ncbi.nlm.nih.gov) -- CID -> InChIKey and name synonyms.
  * **CIViC** (civicdb.org) -- all PREDICTIVE evidence (SENSITIVITY and RESISTANCE): therapy ->
    molecular profile (gene + variant) with evidence level and PubMed ID. A drug is matched to its CIViC
    therapy by any of its dataset names or synonyms (e.g. MK-1775 = adavosertib).
  * **Europe PMC** (ebi.ac.uk) -- for compounds of an indirect-biomarker class (see INDIRECT_MAP) that CIViC
    does not list: a primary paper that names the compound in its title confirms the class.

No per-drug rule is supplied. The script embeds only three small class-level tables of published
mechanism facts: PATHWAY (pathway -> member genes), TSG (canonical tumour suppressors, for the
gene-role-aware state classification) and INDIRECT_MAP (target class -> expected synthetic-lethal /
dependence biomarker + direction: PARP -> homologous-recombination genes LoF; MDM2/MDM4 -> TP53
wildtype; ATR/CHK1/CHK2/WEE1 -> TP53 LoF).

``sensitive_status`` is classified from the CIViC profile text: wildtype -> ``wildtype``; loss of
function / loss / inactivating, or a mutation/deletion/truncation of a tumour suppressor -> ``LoF``;
otherwise ``activating``.

Output columns: drug_name, pubchem_id, datasets, biomarker_gene, sensitive_status, target_genes,
target_source, evidence_type, evidence_level, n_evidence_items, review_items_excluded, source.

Reviews are not used as evidence: the publication type of every cited paper is read from Europe PMC, and
CIViC evidence items whose source is a review are skipped (their number is reported per rule in
``review_items_excluded``); the literature route additionally rejects reviews by publication type.

Run:
  python create_target_mutation_seed.py --data_path data --out target_mutation_seed.csv [--email ADDR]
"""

import argparse
import io
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ZENODO_DOI_URL = "https://zenodo.org/doi/10.5281/zenodo.12633909"
TOY_URL = "https://github.com/nf-core/test-datasets/raw/refs/heads/drugresponseeval/test_data"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
FALLBACK_DATASETS = ["GDSC1", "GDSC2", "CCLE", "TOYv1", "TOYv2", "CTRPv1", "CTRPv2", "BeatAML2", "PDX_Bruna"]

PATHWAY = {
    "RAS_MAPK": {"BRAF", "KRAS", "NRAS", "HRAS", "NF1", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3", "RAF1"},
    "PI3K_AKT": {"PIK3CA", "PIK3CB", "PIK3R1", "AKT1", "AKT2", "AKT3", "PTEN", "MTOR"},
}
# Canonical tumour suppressors (COSMIC Cancer Gene Census, role = TSG): a plain mutation / deletion of one
# of these means loss of function (gene-role-aware state classification).
TSG = {
    "TP53",
    "PTEN",
    "BRCA1",
    "BRCA2",
    "RB1",
    "NF1",
    "STK11",
    "VHL",
    "CDKN2A",
    "SMAD4",
    "APC",
    "ATM",
    "PALB2",
    "ATR",
    "CHEK2",
    "MLH1",
    "MSH2",
    "ARID1A",
    "KEAP1",
    "FBXW7",
    "NF2",
    "TSC1",
    "TSC2",
    "RAD51C",
    "RAD51D",
    "BARD1",
}
# Established synthetic-lethal / dependence principles: target class -> (expected non-target biomarkers,
# direction, biomarkers used for a literature-confirmed rule, Europe PMC class keywords).
# Not included: PI3Kbeta -> PTEN loss. That dependence holds for beta-selective inhibitors only, and ChEMBL
# lists beta-selective and pan-PI3K inhibitors alike at PI3K-family level, so the class cannot be assigned
# from the sources used here.
INDIRECT_MAP = {
    frozenset({"PARP1", "PARP2"}): (
        {"BRCA1", "BRCA2", "PALB2", "ATM", "RAD51C", "RAD51D", "BARD1"},
        "LoF",
        ["BRCA1", "BRCA2"],
        "PARP",
    ),
    frozenset({"MDM2", "MDM4"}): ({"TP53"}, "wildtype", ["TP53"], "MDM2 OR p53"),
    frozenset({"ATR", "CHEK1", "CHEK2", "WEE1"}): ({"TP53"}, "LoF", ["TP53"], "WEE1 OR ATR OR CHK1 OR checkpoint"),
}
STOP = {
    "OR",
    "AND",
    "NOT",
    "TKD",
    "ITD",
    "DNA",
    "WILD",
    "TYPE",
    "LOSS",
    "GAIN",
    "EXON",
    "FUSION",
    "MUTATION",
    "AMPLIFICATION",
    "DELETION",
    "OVEREXPRESSION",
    "EXPRESSION",
    "METHYLATION",
    "PROMOTER",
    "POSITIVE",
    "NEGATIVE",
    "FUNCTION",
    "REARRANGEMENT",
    "V",
}
REVIEW = re.compile(
    r"\b(review|guide to pharmacology|perspective|a primer|: an? overview|: biology|"
    r"current (status|concepts)|advances in|in human disease|in cancer therapy\.?$)\b",
    re.I,
)
SCREEN = re.compile(
    r"genomics of drug sensitivity|cancer cell line encyclopedia|nomenclature|"
    r"\blandscape\b|interactome|data-mining|heterogeneous set",
    re.I,
)


# --------------------------------------------------------------------------------------------- HTTP
class Limiter:
    """Minimum interval between requests to one service (shared by all worker threads)."""

    def __init__(self, interval):
        """
        Set the minimum interval.

        :param interval: minimum number of seconds between two requests
        """
        self.interval, self.lock, self.t = interval, threading.Lock(), 0.0

    def wait(self):
        """Block until the minimum interval since the last request has passed."""
        with self.lock:
            delay = self.t + self.interval - time.time()
            if delay > 0:
                time.sleep(delay)
            self.t = time.time()


PUBCHEM_LIM, CHEMBL_LIM, EPMC_LIM = Limiter(0.22), Limiter(0.1), Limiter(0.1)
FAILED_REQUESTS: list[tuple[str, str]] = []  # (service, url) of requests that still failed after all retries


def get_json(url, limiter=None, data=None, headers=None, retries=5, timeout=45):
    """
    Send a GET or POST request and parse the JSON answer.

    Status 400 and 404 mean that there is no such record and give {}. Other failures are retried with back-off (longer
    for 429 and 5xx) and, if they persist, recorded in FAILED_REQUESTS so that they are reported.

    :param url: URL of the request
    :param limiter: optional Limiter of the service
    :param data: optional request body, which makes the request a POST
    :param headers: optional extra request headers
    :param retries: number of retries after the first attempt
    :param timeout: timeout of one attempt in seconds
    :returns: the parsed JSON, or {} if there is no record or the request failed
    """
    for attempt in range(retries + 1):
        try:
            if limiter:
                limiter.wait()
            hdr = {"Accept": "application/json"}
            hdr.update(headers or {})
            with urllib.request.urlopen(  # noqa: S310
                urllib.request.Request(url, data=data, headers=hdr), timeout=timeout
            ) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                return {}
            time.sleep((5.0 if e.code in (429, 500, 502, 503, 504) else 1.5) * (attempt + 1))
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    FAILED_REQUESTS.append((urllib.parse.urlparse(url).netloc, url[:120]))
    return {}


def norm(name):
    """
    Normalise a name for comparison.

    :param name: drug or therapy name
    :returns: the name in lower case without anything but letters and digits
    """
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# ------------------------------------------------------------------------- drugs of drevalpy datasets
class RangeFile(io.RawIOBase):
    """Read-only seekable file over an HTTP URL (range requests); lets zipfile read one member."""

    def __init__(self, url):
        """
        Open the URL and read its size.

        :param url: URL of a file whose server supports range requests
        """
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
            self.size = int(r.headers["Content-Range"].split("/")[-1])

    def seekable(self):
        """
        Say that the file is seekable.

        :returns: True
        """
        return True

    def readable(self):
        """
        Say that the file is readable.

        :returns: True
        """
        return True

    def tell(self):
        """
        Give the current position.

        :returns: the position in bytes
        """
        return self.pos

    def seek(self, offset, whence=0):
        """
        Move to a position.

        :param offset: offset in bytes
        :param whence: 0 for the start, 1 for the current position, 2 for the end of the file
        :returns: the new position
        """
        self.pos = offset if whence == 0 else (self.pos + offset if whence == 1 else self.size + offset)
        return self.pos

    def read(self, n=-1):
        """
        Read bytes from the current position with a range request.

        :param n: number of bytes, or everything up to the end if negative or None
        :returns: the bytes read
        :raises OSError: if the range request fails four times
        """
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        for attempt in range(4):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
                    chunk = r.read()
                self.pos += len(chunk)
                return chunk
            except Exception:
                time.sleep(2 * (attempt + 1))
        raise OSError("range request failed")

    def readinto(self, buf):
        """
        Read into a buffer.

        :param buf: writable buffer
        :returns: the number of bytes read
        """
        chunk = self.read(len(buf))
        buf[: len(chunk)] = chunk
        return len(chunk)


_ZENODO_FILES: dict[str, str] = {}


def zenodo_files():
    """
    Look up the files of drevalpy's Zenodo record, like drevalpy.datasets.utils.download_dataset.

    :returns: file name and download URL of each file
    """
    if not _ZENODO_FILES:
        for attempt in range(6):
            try:
                with urllib.request.urlopen(ZENODO_DOI_URL, timeout=60) as r:  # noqa: S310
                    link = re.search(r'<([^>]+)>\s*;\s*rel="linkset"', r.headers.get("Link", ""))
                if link:
                    record = get_json(link.group(1))
                    _ZENODO_FILES.update({f["key"]: f["links"]["self"] for f in record.get("files", [])})
                    break
            except Exception:  # noqa: S110
                pass
            time.sleep(4 * (attempt + 1))
    return _ZENODO_FILES


def member_drug_names(zf):
    """
    Read drug_names.csv from a dataset archive.

    :param zf: open zip file
    :returns: the table, or None if the archive has no drug_names.csv
    """
    for member in zf.namelist():
        if member.endswith("drug_names.csv"):
            return pd.read_csv(io.BytesIO(zf.read(member)), dtype=str)
    return None


def dataset_drug_names(dataset, data_path, remote=True):
    """
    Read drug_names.csv (pubchem_id, drug_name) of one drevalpy dataset.

    :param dataset: name of the dataset
    :param data_path: drevalpy data directory
    :param remote: whether to fetch the file if the dataset is not in data_path
    :returns: the table (None if it could not be read) and where it came from
    """
    local = Path(data_path) / dataset / "drug_names.csv"
    if local.exists():
        return pd.read_csv(local, dtype=str), "local"
    if not remote:
        return None, "missing (remote disabled)"
    try:
        files = zenodo_files()
        if not files and not dataset.upper().startswith("TOY"):
            return None, "Zenodo record could not be read"
        if f"{dataset}.zip" in files:
            with zipfile.ZipFile(RangeFile(files[f"{dataset}.zip"])) as zf:
                return member_drug_names(zf), "zenodo"
        if dataset.upper().startswith("TOY"):
            with urllib.request.urlopen(f"{TOY_URL}/{dataset}.zip", timeout=120) as r:  # noqa: S310
                with zipfile.ZipFile(io.BytesIO(r.read())) as zf:
                    return member_drug_names(zf), "nf-core test data"
    except Exception as exc:
        return None, f"failed ({repr(exc)[:80]})"
    return None, "not found"


def default_datasets():
    """
    List the datasets whose drugs are used.

    :returns: the datasets of drevalpy without the derived variants, or a built-in list if drevalpy cannot be imported
    """
    try:
        from drevalpy.datasets.loader import AVAILABLE_DATASETS, DERIVED_DATASETS

        return [d for d in AVAILABLE_DATASETS if d not in DERIVED_DATASETS]
    except Exception:
        return FALLBACK_DATASETS


def collect_drugs(datasets, data_path, remote):
    """
    Collect the drugs of all datasets, one entry per PubChem id.

    :param datasets: names of the datasets
    :param data_path: drevalpy data directory
    :param remote: whether to fetch datasets that are not in data_path
    :returns: names and datasets of each drug by PubChem id, and the number of drugs per dataset
    """
    drugs, per_dataset = {}, {}
    for ds in datasets:
        df, origin = dataset_drug_names(ds, data_path, remote)
        if df is None or "pubchem_id" not in df:
            print(f"  {ds:10s} skipped: {origin}", flush=True)
            continue
        per_dataset[ds] = len(df)
        print(f"  {ds:10s} {len(df):4d} drugs ({origin})", flush=True)
        for pid, name in zip(df["pubchem_id"], df["drug_name"]):
            info = drugs.setdefault(str(pid).strip(), {"names": [], "datasets": []})
            if str(name).strip() not in info["names"]:
                info["names"].append(str(name).strip())
            if ds not in info["datasets"]:
                info["datasets"].append(ds)
    return drugs, per_dataset


# ---------------------------------------------------------------------------- targets (ChEMBL/PubChem)
_TARGET_GENES: dict[str, set[str]] = {}


def target_genes(target_chembl_id):
    """
    Look up the gene symbols of a ChEMBL target.

    :param target_chembl_id: ChEMBL id of the target
    :returns: the gene symbols
    """
    if target_chembl_id not in _TARGET_GENES:
        genes = set()
        record = get_json(f"{CHEMBL}/target/{target_chembl_id}.json", CHEMBL_LIM)
        for comp in record.get("target_components", []):
            for syn in comp.get("target_component_synonyms", []):
                if syn.get("syn_type") == "GENE_SYMBOL":
                    genes.add(syn.get("component_synonym", "").upper())
        _TARGET_GENES[target_chembl_id] = genes
    return _TARGET_GENES[target_chembl_id]


def pubchem_inchikey(cid):
    """
    Look up the InChIKey of a PubChem compound.

    :param cid: PubChem CID
    :returns: the InChIKey, or None
    """
    d = get_json(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/InChIKey/JSON", PUBCHEM_LIM)
    props = d.get("PropertyTable", {}).get("Properties", [])
    return props[0].get("InChIKey") if props else None


def pubchem_synonyms(cid):
    """
    Look up plausible drug-name synonyms of a PubChem compound (IUPAC strings and registry noise filtered out).

    :param cid: PubChem CID
    :returns: the synonyms
    """
    d = get_json(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON", PUBCHEM_LIM)
    try:
        syns = d["InformationList"]["Information"][0].get("Synonym", [])
    except (KeyError, IndexError):
        return []
    return [s for s in syns if re.fullmatch(r"[A-Za-z][A-Za-z0-9 \-]{2,28}", s) and sum(c.isdigit() for c in s) <= 6]


def chembl_names(mol):
    """
    List the names of a ChEMBL molecule.

    :param mol: ChEMBL molecule record
    :returns: preferred name and synonyms
    """
    names = [mol.get("pref_name") or ""] + [s.get("molecule_synonym", "") for s in (mol.get("molecule_synonyms") or [])]
    return [n for n in names if n]


def chembl_molecule(cid, names):
    """
    Find the ChEMBL molecule by the InChIKey of the PubChem CID, else by exact (normalised) name.

    :param cid: PubChem CID or code of the drug
    :param names: names of the drug
    :returns: the molecule record, or None
    """
    if cid.isdigit():
        key = pubchem_inchikey(cid)
        if key:
            mols = get_json(f"{CHEMBL}/molecule.json?molecule_structures__standard_inchi_key={key}", CHEMBL_LIM).get(
                "molecules", []
            )
            if mols:
                return mols[0]
    for name in names[:3]:
        if len(norm(name)) < 4:
            continue
        for mol in get_json(f"{CHEMBL}/molecule/search.json?q={urllib.parse.quote(name)}&limit=5", CHEMBL_LIM).get(
            "molecules", []
        ):
            if norm(name) in {norm(n) for n in chembl_names(mol)}:
                return mol
    return None


def mechanism_genes(molecule_ids):
    """
    Look up the target genes of the mechanism-of-action records.

    ChEMBL often attaches the record to a salt or form of the drug whose parent is the molecule asked for, so both the
    direct and the parent filter are queried.

    :param molecule_ids: ChEMBL ids of the molecule and its parent
    :returns: the target genes
    """
    genes = set()
    for mid in molecule_ids:
        for flt in ("molecule_chembl_id", "parent_molecule_chembl_id"):
            for mech in get_json(f"{CHEMBL}/mechanism.json?{flt}={mid}&limit=50", CHEMBL_LIM).get("mechanisms", []):
                if mech.get("target_chembl_id"):
                    genes |= target_genes(mech["target_chembl_id"])
    return genes


def activity_genes(molecule_ids, min_pchembl):
    """
    Look up the targets with potent binding-assay activity (human, assay type B).

    Targets with at least half as many measurements as the most-measured target are kept.

    :param molecule_ids: ChEMBL ids of the molecule and its parent
    :param min_pchembl: potency cutoff (pChEMBL)
    :returns: the target genes
    """
    counts = {}
    for mid in molecule_ids:
        url = (
            f"{CHEMBL}/activity.json?molecule_chembl_id={mid}&pchembl_value__gte={min_pchembl}"
            f"&assay_type=B&target_organism=Homo%20sapiens&limit=500"
        )
        for act in get_json(url, CHEMBL_LIM).get("activities", []):
            if act.get("target_chembl_id"):
                counts[act["target_chembl_id"]] = counts.get(act["target_chembl_id"], 0) + 1
    if not counts:
        return set()
    top = max(counts.values())
    genes = set()
    for tid, n in counts.items():
        if n >= 0.5 * top:
            genes |= target_genes(tid)
    return genes


def resolve_drug(cid, names, min_pchembl):
    """
    Find the target genes of one drug and extra names (ChEMBL and PubChem synonyms).

    :param cid: PubChem CID or code of the drug
    :param names: names of the drug
    :param min_pchembl: potency cutoff (pChEMBL) of the binding-activity fallback
    :returns: the target genes, how they were found (``genes``, ``how``) and the synonyms (``syns``)
    """
    out = {"genes": set(), "how": "", "syns": []}
    if cid.isdigit():
        out["syns"] += pubchem_synonyms(cid)
    mol = chembl_molecule(cid, names)
    if mol:
        out["syns"] += chembl_names(mol)
        ids = {mol["molecule_chembl_id"], (mol.get("molecule_hierarchy") or {}).get("parent_chembl_id")} - {None}
        out["genes"] = mechanism_genes(ids)
        out["how"] = "chembl_mechanism" if out["genes"] else ""
        if not out["genes"]:
            out["genes"] = activity_genes(ids, min_pchembl)
            out["how"] = "chembl_binding_activity" if out["genes"] else ""
    return out


# ------------------------------------------------------------------------------------------- CIViC
def civic(query):
    """
    Send a GraphQL query to CIViC.

    :param query: GraphQL query
    :returns: the parsed answer
    """
    return get_json(
        "https://civicdb.org/api/graphql",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json"},
    )


def pull_civic(significance):
    """
    Pull all PREDICTIVE evidence of one significance from CIViC.

    :param significance: SENSITIVITYRESPONSE or RESISTANCE
    :returns: therapy, molecular profile, evidence level, evidence id and PubMed id of each evidence item
    """
    out, cursor = [], None
    while True:
        after = f', after:"{cursor}"' if cursor else ""
        q = (
            "{ evidenceItems(evidenceType:PREDICTIVE, significance:%s, first:100%s){ "
            "pageInfo{ hasNextPage endCursor } nodes{ id evidenceLevel status therapies{ name } "
            "molecularProfile{ name } source{ citationId } } } }"
        ) % (significance, after)
        ei = (civic(q).get("data") or {}).get("evidenceItems") or {}
        for n in ei.get("nodes", []):
            if n.get("status") == "REJECTED":
                continue
            prof = (n.get("molecularProfile") or {}).get("name", "")
            src = (n.get("source") or {}).get("citationId", "")
            for t in n.get("therapies") or []:
                out.append((t.get("name", ""), prof, n.get("evidenceLevel"), n.get("id"), src))
        info = ei.get("pageInfo", {})
        if not info.get("hasNextPage"):
            break
        cursor = info.get("endCursor")
    return out


def classify(profile, gene=None):
    """
    Classify the sensitive state from the text of a CIViC profile, aware of the gene role.

    :param profile: name of the molecular profile
    :param gene: gene of the biomarker
    :returns: ``wildtype``, ``LoF`` or ``activating``
    """
    p = profile.lower()
    if "wildtype" in p or "wild-type" in p or "wild type" in p:
        return "wildtype"
    if (
        "loss-of-function" in p
        or "loss of function" in p
        or re.search(r"\bloss\b", p)
        or "biallelic inactivation" in p
        or "inactivating" in p
    ):
        return "LoF"
    if gene in TSG and ("mutation" in p or "deletion" in p or "frameshift" in p or "truncat" in p):
        return "LoF"
    return "activating"


def genes_in(profile):
    """
    Find the gene symbols in the name of a molecular profile.

    :param profile: name of the molecular profile
    :returns: the gene symbols
    """
    toks = re.findall(r"[A-Z][A-Z0-9]{1,}", profile.replace("::", " "))
    return [t for t in toks if t not in STOP and not t.isdigit()]


def publication_types(pmids, email=None):
    """
    Read the publication types from Europe PMC (core records), queried in batches.

    :param pmids: PubMed ids
    :param email: optional contact address passed to Europe PMC
    :returns: the publication types in lower case by PubMed id; ids that Europe PMC does not return are absent
    """
    out, pmids = {}, list(pmids)
    for i in range(0, len(pmids), 40):
        chunk = pmids[i : i + 40]  # noqa: E203
        q = "(" + " OR ".join(f"EXT_ID:{p}" for p in chunk) + ") AND SRC:MED"
        url = (
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query="
            + urllib.parse.quote(q)
            + f"&format=json&resultType=core&pageSize={len(chunk) + 5}"
        )
        if email:
            url += "&email=" + urllib.parse.quote(email)
        for x in get_json(url, EPMC_LIM).get("resultList", {}).get("result", []):
            if x.get("pmid"):
                out[x["pmid"]] = [t.lower() for t in (x.get("pubTypeList") or {}).get("pubType", [])]
    return out


def is_review(types):
    """
    Check for a review among Europe PMC publication types ('Review', 'review-article', 'Systematic Review').

    :param types: publication types in lower case
    :returns: True if any is a review
    """
    return any("review" in t for t in types)


def epmc_named(drug, keywords, email=None):
    """
    Find a primary paper that names the compound in its title (no review or screen paper).

    Reviews are excluded by Europe PMC publication type and, in addition, by title pattern.

    :param drug: name of the compound
    :param keywords: Europe PMC search terms of the target class
    :param email: optional contact address passed to Europe PMC
    :returns: the citation of the most cited paper, or None
    """
    q = f'"{drug}" AND ({keywords})'
    url = (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query="
        + urllib.parse.quote(q)
        + "&format=json&pageSize=30&resultType=core&sort=CITED%20desc"
    )
    if email:
        url += "&email=" + urllib.parse.quote(email)
    best, token = None, norm(drug)
    for x in get_json(url, EPMC_LIM).get("resultList", {}).get("result", []):
        title = x.get("title", "")
        if x.get("source") != "MED" or not x.get("pmid") or token not in norm(title):
            continue
        if is_review([t.lower() for t in (x.get("pubTypeList") or {}).get("pubType", [])]):
            continue
        if REVIEW.search(title) or SCREEN.search(title):
            continue
        cited = int(x.get("citedByCount", 0))
        if best is None or cited > best[0]:
            best = (cited, x["pmid"], x.get("pubYear"), title)
    return None if best is None else f"EuropePMC PMID:{best[1]} — {best[3]} ({best[2]})"


def drug_rules(key, info, r, matched, smap, rmap, review_pmids, pathway_of, email):
    """
    Derive the rules (biomarker gene and sensitive state) of one drug.

    :param key: PubChem id of the drug
    :param info: names and datasets of the drug, as returned by collect_drugs
    :param r: targets and synonyms of the drug, as returned by resolve_drug
    :param matched: name under which the drug is listed in CIViC, or None
    :param smap: CIViC sensitivity evidence per normalised therapy name
    :param rmap: CIViC resistance evidence ids per normalised therapy name
    :param review_pmids: PubMed ids whose publication type is a review
    :param pathway_of: pathway bucket of each gene
    :param email: optional contact address passed to Europe PMC
    :returns: the rules as a list of rows, the number of review items skipped and the number of target-gene rules lost
        because all their evidence came from reviews
    """
    rows = []
    n_review_items = n_review_only_rules = 0
    name, targets = info["names"][0], r["genes"]
    syns = [s for s in r["syns"] if len(norm(s)) >= 4]
    indirect_cls = next((c for c in INDIRECT_MAP if targets & c), None)
    indirect = INDIRECT_MAP[indirect_cls] if indirect_cls else None

    def add(gene, status, etype, src, level="", n_items=1, review_items=0):
        rows.append(
            dict(
                drug_name=name,
                pubchem_id=key,
                datasets=";".join(info["datasets"]),
                biomarker_gene=gene,
                sensitive_status=status,
                target_genes=";".join(sorted(targets)),
                target_source=r["how"],
                evidence_type=etype,
                evidence_level=level,
                n_evidence_items=n_items,
                review_items_excluded=review_items,
                source=src,
            )
        )

    per_gene, emitted_indirect, rev_excl = {}, False, {}
    if matched:
        for prof, lvl, eid, pmid in smap[norm(matched)]:
            # fusion profile "A::B": B (3') is the activated gene; the 5' partner is not a biomarker
            # each gene counts once per evidence item, however often it occurs in the profile name
            for g in dict.fromkeys(genes_in(prof.split("::")[-1] if "::" in prof else prof)):
                if str(pmid) in review_pmids:  # source is a review: not used as evidence
                    rev_excl[g] = rev_excl.get(g, 0) + 1
                    n_review_items += 1
                    continue
                e = per_gene.setdefault(g, {"status": {}, "eids": [], "pmids": set(), "levels": set()})
                st = classify(prof, g)
                e["status"][st] = e["status"].get(st, 0) + 1
                e["eids"].append(eid)
                e["pmids"].add(str(pmid))
                e["levels"].add(lvl)

    def emit(gene, etype, force_status=None):
        e = per_gene[gene]
        status = force_status or max(e["status"], key=e["status"].get)
        src = (
            f"CIViC predictive SENSITIVITY ({gene}; "
            + ",".join(f"EID{x}" for x in e["eids"][:5])
            + "; level "
            + sorted(x for x in e["levels"] if x)[0]
            + "; PMID "
            + ",".join(sorted(p for p in e["pmids"] if p and p != "None")[:5])
            + ")"
        )
        if matched != name:
            src = f"[matched via name '{matched}'] " + src
        if norm(matched) in rmap:
            src += f" | CIViC RESISTANCE evidence exists ({len(rmap[norm(matched)])} items)"
        add(
            gene,
            status,
            etype,
            src,
            level=sorted(x for x in e["levels"] if x)[0],
            n_items=len(e["eids"]),
            review_items=rev_excl.get(gene, 0),
        )

    # genes whose only CIViC evidence for this drug came from reviews: no rule (count them)
    n_review_only_rules += sum(1 for g in rev_excl if g not in per_gene and g in targets)

    if per_gene:
        # direct: the biomarker is one of the drug's own target genes (most evidence), else a gene of the
        # same pathway bucket as a target
        direct = sorted(g for g in targets if g in per_gene)
        if direct:
            for g in direct:  # every target gene with CIViC SENSITIVITY evidence for this drug
                emit(g, "civic_predictive_target")
        else:
            buckets = {pathway_of[g] for g in targets if g in pathway_of}
            cands = [g for g in per_gene if pathway_of.get(g) in buckets]
            if cands:
                emit(max(cands, key=lambda g: len(per_gene[g]["eids"])), "civic_predictive_pathway")
        # indirect, confirmed by CIViC evidence of the expected direction for this drug
        if indirect:
            for g in sorted(indirect[0]):  # a set of strings: sort, or the row order depends on PYTHONHASHSEED
                if g in per_gene and indirect[1] in per_gene[g]["status"]:
                    emit(g, "civic_predictive_indirect", force_status=indirect[1])
                    emitted_indirect = True

    # indirect, drug not confirmed by CIViC: class confirmed by a primary paper naming the compound
    if indirect and not emitted_indirect:
        lit = None
        for nm in info["names"] + syns[:8]:
            lit = epmc_named(nm, indirect[3], email)
            if lit:
                break
        if lit:
            for g in indirect[2]:
                add(
                    g,
                    indirect[1],
                    "mechanism_literature",
                    f"mechanism principle (target {'/'.join(sorted(targets & indirect_cls))}"
                    f" -> {g} {indirect[1]}) confirmed by compound characterization: {lit}",
                )
    return rows, n_review_items, n_review_only_rules


# -------------------------------------------------------------------------------------------- main
def main(argv=None):
    """
    Create target_mutation_seed.csv from the drugs of the drevalpy datasets.

    :param argv: command line arguments, ``sys.argv[1:]`` if None
    """
    ap = argparse.ArgumentParser(description="Create target_mutation_seed.csv from the drugs of the drevalpy datasets.")
    ap.add_argument("--data_path", default="data", help="drevalpy data directory (default: data)")
    ap.add_argument("--datasets", nargs="*", default=None, help="datasets to use (default: all drevalpy datasets)")
    ap.add_argument("--out", default="target_mutation_seed.csv", help="output file")
    ap.add_argument("--no_remote", action="store_true", help="read only datasets present in --data_path")
    ap.add_argument(
        "--min_pchembl", type=float, default=7.0, help="potency cutoff of the binding-activity target fallback"
    )
    ap.add_argument("--workers", type=int, default=3, help="parallel target lookups")
    ap.add_argument("--max_drugs", type=int, default=None, help="process only the first N drugs (testing)")
    ap.add_argument("--email", default=None, help="optional contact address passed to Europe PMC")
    a = ap.parse_args(argv)

    # 1) drugs of the drevalpy datasets
    datasets = a.datasets or default_datasets()
    print("datasets:", ", ".join(datasets), flush=True)
    drugs, per_dataset = collect_drugs(datasets, a.data_path, remote=not a.no_remote)
    keys = list(drugs)[: a.max_drugs] if a.max_drugs else list(drugs)
    print(f"{len(keys)} distinct drugs (by PubChem id) from {len(per_dataset)} datasets", flush=True)

    # 2) target genes (ChEMBL) and synonyms (PubChem/ChEMBL), in parallel
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(lambda k: resolve_drug(k, drugs[k]["names"], a.min_pchembl), keys))
    resolved = dict(zip(keys, results))
    n_mech = sum(r["how"] == "chembl_mechanism" for r in results)
    n_act = sum(r["how"] == "chembl_binding_activity" for r in results)
    print(
        f"targets found: {n_mech} via ChEMBL mechanism, {n_act} via binding activity, "
        f"{len(keys) - n_mech - n_act} without target",
        flush=True,
    )

    # 3) CIViC predictive evidence
    sens, res = pull_civic("SENSITIVITYRESPONSE"), pull_civic("RESISTANCE")
    print(f"CIViC predictive evidence: {len(sens)} SENSITIVITY, {len(res)} RESISTANCE", flush=True)
    smap, rmap = {}, {}
    for therapy, prof, lvl, eid, pmid in sens:
        smap.setdefault(norm(therapy), []).append((prof, lvl, eid, pmid))
    for therapy, _prof, _lvl, eid, _pmid in res:
        rmap.setdefault(norm(therapy), []).append(eid)

    # 3b) publication type of every paper cited by the CIViC evidence of the matched therapies;
    # evidence whose source is a review is not used (reviews are secondary literature)
    matched_of = {}
    for key in keys:
        info, r = drugs[key], resolved[key]
        syns = [x for x in r["syns"] if len(norm(x)) >= 4]
        matched_of[key] = next((n for n in info["names"] + syns if norm(n) in smap), None)
    needed = sorted(
        {
            str(item[3])
            for key in keys
            if matched_of[key]
            for item in smap[norm(matched_of[key])]
            if str(item[3]).isdigit()
        }
    )
    ptypes = publication_types(needed, a.email)
    review_pmids = {p for p, t in ptypes.items() if is_review(t)}
    print(
        f"cited PMIDs: {len(needed)}; publication type known for {len(ptypes)}; reviews: {len(review_pmids)} "
        f"(their evidence items are not used); unverifiable (not in Europe PMC): {len(needed) - len(ptypes)}",
        flush=True,
    )

    # 4) rules
    pathway_of = {g: b for b, members in PATHWAY.items() for g in members}
    rows = []
    n_review_items = n_review_only_rules = 0
    for key in keys:
        drug_rows, n_items, n_only = drug_rules(
            key, drugs[key], resolved[key], matched_of[key], smap, rmap, review_pmids, pathway_of, a.email
        )
        rows += drug_rows
        n_review_items += n_items
        n_review_only_rules += n_only

    columns = [
        "drug_name",
        "pubchem_id",
        "datasets",
        "biomarker_gene",
        "sensitive_status",
        "target_genes",
        "target_source",
        "evidence_type",
        "evidence_level",
        "n_evidence_items",
        "review_items_excluded",
        "source",
    ]
    out = pd.DataFrame(rows, columns=columns).drop_duplicates(subset=["pubchem_id", "biomarker_gene"])
    out.to_csv(a.out, index=False)
    print(f"\nrules: {len(out)} for {out.pubchem_id.nunique()} drugs (of {len(keys)})")
    if FAILED_REQUESTS:
        by_service = pd.Series([f[0] for f in FAILED_REQUESTS]).value_counts().to_dict()
        print(
            f"WARNING: {len(FAILED_REQUESTS)} requests failed after retries {by_service}; "
            "affected drugs may be missing rules - re-run to complete"
        )
    else:
        print("all web requests succeeded")
    print(
        f"review-sourced CIViC evidence not used: {n_review_items} (evidence item x gene); "
        f"target-gene rules lost because all their evidence came from reviews: {n_review_only_rules}"
    )
    print("evidence_level:", out.evidence_level.replace("", "-").value_counts().sort_index().to_dict())
    print("evidence_type:", out.evidence_type.value_counts().to_dict())
    print("status:", out.sensitive_status.value_counts().to_dict())
    print("written:", a.out)


if __name__ == "__main__":
    main()
