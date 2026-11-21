"""Tests for fitting replicates on their own and building the replicate-fits table."""

from __future__ import annotations

import anndata
import numpy as np
import pandas as pd
import pytest

from drevalpy.curation._fit import FitOptions, Status, fit_table
from drevalpy.curation._replicate_fits import (
    ID_COLUMNS,
    METRICS,
    REPLICATE_TAG,
    Source,
    build,
    tag_replicates,
    to_schema,
    untag_replicates,
)
from drevalpy.curation._series import SERIES_COLUMNS
from drevalpy.errors import InvalidInputError


def _series(rows: list[tuple]) -> pd.DataFrame:
    """Series rows from ``(sample, drug, replicate, eligible)`` tuples, with one metadata column."""
    frame = pd.DataFrame(rows, columns=pd.Index(["native_sample", "native_drug", "replicate", "eligible"]))
    frame["series_key"] = "P" + frame["replicate"].astype(str)
    frame["experiment"] = "e"
    frame["biological_replicate"] = 0
    frame["technical_replicate"] = frame["replicate"]
    frame["n_doses"] = np.where(frame["eligible"], 8, 2)
    frame["barcode"] = "B" + frame["replicate"].astype(str)
    return frame[[*SERIES_COLUMNS, "barcode"]]


def _fit_row(sample: str, drug: str, pec50: float | None) -> dict:
    row: dict = {"sample": sample, "drug": drug}
    row |= {metric: np.nan for metric in METRICS}
    row["regulation"] = "down" if pec50 is not None else np.nan
    row["status"] = Status.FITTED.value if pec50 is not None else Status.FIT_FAILED.value
    row["pEC50"] = np.nan if pec50 is None else pec50
    row["AUC"] = np.nan if pec50 is None else pec50 * 10
    return row


def _own(rows: list[tuple]) -> pd.DataFrame:
    """Own fits from ``(sample, drug, replicate, pEC50 or None)`` tuples."""
    out = [{**_fit_row(sample, drug, pec50), "replicate": replicate} for sample, drug, replicate, pec50 in rows]
    return pd.DataFrame(out, columns=pd.Index(["sample", "drug", "replicate", "status", *METRICS]))


def _pooled(values: dict[tuple[str, str], float | None]) -> pd.DataFrame:
    rows = [_fit_row(sample, drug, pec50) for (sample, drug), pec50 in values.items()]
    return pd.DataFrame(rows, columns=pd.Index(["sample", "drug", "status", *METRICS]))


class TestBuild:
    def test_multi_series_pairs_carry_their_own_fits(self) -> None:
        series = _series([("c", "d", 0, True), ("c", "d", 1, True)])
        table = build(_own([("c", "d", 0, 5.0), ("c", "d", 1, 6.0)]), _pooled({("c", "d"): 9.0}), series)
        assert table["pEC50"].tolist() == [5.0, 6.0]
        assert set(table["source"]) == {Source.REPLICATE}
        assert set(table["status"]) == {Status.FITTED}

    def test_a_single_series_pair_gets_the_pooled_fit(self) -> None:
        series = _series([("c", "d", 0, True), ("c", "d", 1, False)])
        table = build(_own([]), _pooled({("c", "d"): 9.0}), series)
        assert table["source"].tolist() == [Source.POOLED, Source.REPLICATE]
        assert table["status"].tolist() == [Status.FITTED, Status.TOO_FEW_DOSES]
        assert table.loc[0, "pEC50"] == 9.0
        assert table.loc[0, "AUC"] == 90.0
        assert np.isnan(table.loc[1, "pEC50"])

    def test_a_dropped_fit_is_failed(self) -> None:
        series = _series([("c", "d", 0, True), ("c", "d", 1, True)])
        table = build(_own([("c", "d", 0, 5.0)]), _pooled({}), series)
        assert table["status"].tolist() == [Status.FITTED, Status.FIT_FAILED]

    @pytest.mark.parametrize("pooled", [{("x", "y"): 1.0}, {("c", "d"): None}], ids=["absent", "failed"])
    def test_a_pooled_fit_that_failed_is_failed(self, pooled: dict) -> None:
        table = build(_own([]), _pooled(pooled), _series([("c", "d", 0, True)]))
        assert table["status"].tolist() == [Status.FIT_FAILED]

    def test_regulation_is_encoded_like_the_response_layer(self) -> None:
        from drevalpy.curation._anndata import _REGULATION_ENCODING

        series = _series([("c", "d", 0, True), ("c", "d", 1, True)])
        table = build(_own([("c", "d", 0, 5.0), ("c", "d", 1, 6.0)]), _pooled({}), series)
        assert table["regulation"].tolist() == [_REGULATION_ENCODING["down"]] * 2

    def test_metadata_columns_follow_the_metrics(self) -> None:
        table = build(_own([]), _pooled({("c", "d"): 1.0}), _series([("c", "d", 0, True)]))
        assert list(table.columns)[-1] == "barcode"

    def test_the_replicate_hierarchy_is_carried_through(self) -> None:
        series = _series([("c", "d", 0, True), ("c", "d", 1, True)])
        series["biological_replicate"] = [0, 1]
        series["technical_replicate"] = [0, 0]
        table = build(_own([("c", "d", 0, 5.0), ("c", "d", 1, 6.0)]), _pooled({}), series)
        assert table[["biological_replicate", "technical_replicate"]].values.tolist() == [[0, 0], [1, 0]]


def test_the_metrics_are_x_and_the_layers_of_build_anndata() -> None:
    from drevalpy.curation._anndata import _LAYER_METRICS

    assert METRICS == ["pEC50", *_LAYER_METRICS]


def _table() -> pd.DataFrame:
    table = build(
        _own([]),
        _pooled({("c1", "d1"): 1.0, ("c2", "d2"): 2.0}),
        _series([("c1", "d1", 0, True), ("c2", "d2", 0, True), ("c1", "d2", 0, False)]),
    )
    table.insert(0, "pubchem_id", table["native_drug"])
    table.insert(0, "sample_id", table["native_sample"])
    return to_schema(table)


class TestSchema:
    def test_columns_are_ordered(self) -> None:
        assert list(_table().columns) == [*ID_COLUMNS, *METRICS, "barcode"]

    def test_dtypes(self) -> None:
        table = _table()
        assert table["replicate"].dtype == np.int32
        assert table["biological_replicate"].dtype == np.int32
        assert table["technical_replicate"].dtype == np.int32
        assert table["n_doses"].dtype == np.int16
        assert (table[METRICS].dtypes == np.float32).all()
        assert isinstance(table["status"].dtype, pd.CategoricalDtype)

    def test_the_table_round_trips_through_h5ad_uns(self, tmp_path) -> None:
        table = _table()
        table.loc[0, "barcode"] = None
        adata = anndata.AnnData(X=np.zeros((2, 2), dtype=np.float32), uns={"replicate_fits": table})
        path = tmp_path / "r.h5ad"
        adata.write_h5ad(path)
        back = anndata.read_h5ad(path).uns["replicate_fits"]
        pd.testing.assert_frame_equal(back.reset_index(drop=True), table, check_categorical=False)


class TestTagging:
    @staticmethod
    def _viability(sample: str = "A549", drug: str = "D1") -> pd.DataFrame:
        doses = [0.01, 0.1, 1.0, 10.0, 100.0]
        rows = [
            {"sample": s, "drug": d, "concentration": dose, "intensity": 1 / (1 + dose), "replicate": r}
            for s, d, replicates in ((sample, drug, (0, 1)), ("HeLa", "D2", (0,)))
            for r in replicates
            for dose in doses
        ]
        return pd.DataFrame(rows)

    def test_a_label_containing_the_tag_is_rejected(self) -> None:
        with pytest.raises(InvalidInputError, match="sample 'A549__rep_3'"):
            tag_replicates(self._viability(sample=f"A549{REPLICATE_TAG}3"))

    def test_single_replicate_pairs_are_left_out(self) -> None:
        viability = self._viability()

        tagged = tag_replicates(viability[viability["sample"] == "HeLa"])

        assert tagged.empty
        assert "replicate" not in tagged.columns

    def test_no_fit_decodes_to_an_empty_table(self) -> None:
        fits = untag_replicates(fit_table(tag_replicates(self._viability().iloc[:0]), FitOptions()))

        assert fits.empty
        assert list(fits.columns) == ["sample", "drug", "replicate", "status"]

    @pytest.mark.slow
    def test_each_replicate_is_fitted_and_decoded_back(self) -> None:
        fits = untag_replicates(fit_table(tag_replicates(self._viability()), FitOptions(fit_speed="fast")))

        assert fits[["sample", "drug", "replicate"]].values.tolist() == [["A549", "D1", 0], ["A549", "D1", 1]]
        assert set(fits["status"]) == {Status.FITTED}
