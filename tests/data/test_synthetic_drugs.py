"""Tests for :mod:`drevalpy.data._synthetic_drugs`, the drug-side half of ``synthetic``."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.data._synthetic_drugs import (
    BPE_LENGTH,
    CHEMBERTA_DIM,
    DRUG_VIEWS,
    FINGERPRINT_BITS,
    MAX_SYNTHETIC_DRUGS,
    SMILES,
    SMILESVEC_DIM,
    UNS_PAYLOADS,
    _bpe_codes,
    _graph_to_dict,
    _pathways_gmt,
    _sparsego_uns,
    add_drug_views,
    add_uns_payloads,
)
from drevalpy.testing import build_synthetic_dataset

pytestmark = pytest.mark.slow

GENES = [f"G{index}" for index in range(8)]


def test_every_smiles_parses_and_is_distinct_and_the_limit_is_the_molecule_count():
    from rdkit import Chem

    assert MAX_SYNTHETIC_DRUGS == len(SMILES)
    assert len(set(SMILES)) == len(SMILES)
    assert all(Chem.MolFromSmiles(value) is not None for value in SMILES)


class TestTextPayloads:
    @pytest.mark.parametrize(("genes", "n_per_set"), [(GENES, 5), (GENES[:3], 3)])
    def test_the_gmt_has_two_sets_of_up_to_five_genes(self, genes, n_per_set):
        rows = [line.split("\t") for line in _pathways_gmt(genes).splitlines()]

        assert [row[0] for row in rows] == ["SYNTH_PATHWAY_A", "SYNTH_PATHWAY_B"]
        assert all(len(row) - 2 >= n_per_set for row in rows)

    def test_sparsego_indexes_every_gene_and_links_it_to_a_term(self):
        payload = _sparsego_uns(GENES)

        assert payload["gene2ind"].splitlines()[2] == "2\tG2"
        gene_rows = [line for line in payload["ontology"].splitlines() if line.endswith("\tgene")]
        assert [row.split("\t")[1] for row in gene_rows] == GENES

    def test_bpe_codes_learn_merges_from_the_smiles(self):
        codes = _bpe_codes(SMILES[:8], num_symbols=10)

        assert codes.startswith("#version")
        assert len(codes.splitlines()) > 1


def test_graph_to_dict_stores_plain_typed_arrays():
    class Graph:
        x = [[1.0, 2.0]]
        edge_index = [[0], [0]]
        edge_attr = [[1.0]]

    stored = _graph_to_dict(Graph())

    assert {key: value.dtype for key, value in stored.items()} == {
        "x": np.float32,
        "edge_index": np.int64,
        "edge_attr": np.float32,
    }


class TestAddViewsAndPayloads:
    def test_drug_views_are_filled_at_their_width(self):
        data = build_synthetic_dataset(n_drugs=8)

        add_drug_views(data, DRUG_VIEWS, np.random.default_rng(0))

        varm = data.mdata.mod["response"].varm
        assert {name: varm[name].shape[1] for name in DRUG_VIEWS} == {
            "morgan_fingerprint": FINGERPRINT_BITS,
            "chemberta": CHEMBERTA_DIM,
            "bpe_smiles": BPE_LENGTH,
            "smilesvec": SMILESVEC_DIM,
        }

    def test_uns_payloads_are_all_filled(self):
        data = build_synthetic_dataset(n_drugs=8)

        add_uns_payloads(data, UNS_PAYLOADS, GENES)

        assert set(UNS_PAYLOADS) <= set(data.mdata.uns)
        assert len(data.mdata.uns["drug_graphs"]) == 8

    def test_unknown_views_and_payloads_are_rejected(self):
        with pytest.raises(ValueError, match="drug_views"):
            add_drug_views(build_synthetic_dataset(), ["nope"], np.random.default_rng(0))
        with pytest.raises(ValueError, match="uns_payloads"):
            add_uns_payloads(build_synthetic_dataset(), ["nope"], GENES)
