"""Tests for grouping wells into numbered dose series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drevalpy.curation._series import SERIES_COLUMNS, VIABILITY_COLUMNS, Series, number_series
from drevalpy.errors import InvalidInputError

METADATA = ("plate", "DATE")


def _wells(*series: tuple) -> pd.DataFrame:
    """Rows from ``(sample, drug, replicate, experiment, doses, date)`` tuples, one well per dose."""
    rows = []
    for sample, drug, replicate, experiment, doses, date in series:
        for position, dose in enumerate(doses):
            rows.append(
                {
                    "sample": sample,
                    "drug": drug,
                    "replicate": replicate,
                    "experiment": experiment,
                    "concentration": dose,
                    "intensity": 0.5,
                    "position": position,
                    "plate": replicate,
                    "DATE": date,
                }
            )
    return pd.DataFrame(rows)


FIVE = [0.01, 0.1, 1.0, 10.0, 100.0]


def _number(wells: pd.DataFrame, metadata: tuple[str, ...] = METADATA, min_doses: int = 4) -> Series:
    return number_series(wells, metadata=metadata, min_doses=min_doses)


def _only_series_key(wells: pd.DataFrame) -> pd.DataFrame:
    """Drop ``experiment`` and ``position``, keeping ``replicate`` as the series key."""
    return wells.drop(columns=["experiment", "position"])


class TestSeries:
    def test_each_plate_is_one_series(self) -> None:
        viability, series = _number(_wells(("c", "d", "P1", "e", FIVE, "x"), ("c", "d", "P2", "e", FIVE, "x")))
        assert series["series_key"].tolist() == ["P1", "P2"]
        assert viability.groupby("replicate").size().tolist() == [5, 5]

    def test_a_missing_well_does_not_move_wells_between_plates(self) -> None:
        """Counting per dose would put P2's wells into replicate 0 where P1 lacks a dose."""
        wells = _wells(("c", "d", "P1", "e", FIVE[:4], "x"), ("c", "d", "P2", "e", FIVE, "x"))
        viability, _ = _number(wells)
        assert viability.groupby("replicate")["concentration"].apply(list).tolist() == [FIVE[:4], FIVE]

    def test_the_output_columns_follow_the_contract(self) -> None:
        viability, series = _number(_wells(("c", "d", "P1", "e", FIVE, "x")))
        assert list(viability.columns) == VIABILITY_COLUMNS
        assert list(series.columns) == [*SERIES_COLUMNS, "plate", "date"]

    def test_a_missing_replicate_label_is_its_own_empty_series(self) -> None:
        wells = _wells(("c", "d", "P1", "e", FIVE, "x"), ("c", "d", None, "e", FIVE, "x"))
        _, series = _number(wells)
        assert series["series_key"].tolist() == ["", "P1"]

    def test_numeric_series_labels_lose_their_decimal_part(self) -> None:
        _, series = _number(_wells(("c", "d", 7.0, 2.0, FIVE, "x")))
        assert series[["series_key", "experiment"]].values.tolist() == [["7", "2"]]

    def test_identifiers_are_native_labels(self) -> None:
        viability, series = _number(_wells(("c", 1001.0, "P1", "e", FIVE, "x")))
        assert set(viability["drug"]) == {"1001"}
        assert series["native_drug"].tolist() == ["1001"]

    def test_numbering_restarts_per_pair_and_follows_the_key(self) -> None:
        wells = _wells(
            ("c", "d", "P2", "e", FIVE, "x"),
            ("c", "d", "P1", "e", FIVE, "x"),
            ("c2", "d", "P9", "e", FIVE, "x"),
        )
        _, series = _number(wells)
        assert series[["native_sample", "series_key", "replicate"]].values.tolist() == [
            ["c", "P1", 0],
            ["c", "P2", 1],
            ["c2", "P9", 0],
        ]

    @pytest.mark.parametrize("column", ["sample", "drug", "concentration", "intensity"])
    def test_a_missing_required_value_drops_the_well(self, column: str) -> None:
        wells = _wells(("c", "d", "P1", "e", FIVE, "x"))
        wells.loc[0, column] = None
        viability, _ = _number(wells)
        assert len(viability) == 4

    def test_concentrations_are_rounded(self) -> None:
        wells = _wells(("c", "d", "P1", "e", [0.1 + 1e-15, 0.1, 1.0, 10.0, 100.0], "x"))
        viability, _ = _number(wells)
        assert viability["concentration"].nunique() == 4

    def test_without_a_replicate_column_every_pair_is_one_series(self) -> None:
        wells = _wells(("c", "d", "P1", "e", FIVE, "x"), ("c", "d", "P2", "e", FIVE, "x"))
        _, series = _number(wells.drop(columns=["replicate", "experiment", "position"]))
        assert len(series) == 1
        assert series.loc[0, "n_doses"] == 5


class TestRepeats:
    def test_a_complete_repeat_block_is_split_by_position(self) -> None:
        wells = _wells(("c", "d", "P1", "e", FIVE + FIVE, "x"))
        wells["intensity"] = [0.1] * 5 + [0.9] * 5
        viability, series = _number(wells)
        assert series["series_key"].tolist() == ["P1#0", "P1#1"]
        assert viability.groupby("replicate")["intensity"].mean().tolist() == pytest.approx([0.1, 0.9])

    def test_a_partial_repeat_is_averaged(self) -> None:
        wells = _wells(("c", "d", "P1", "e", [*FIVE, FIVE[0]], "x"))
        wells["intensity"] = [0.2, 0.5, 0.5, 0.5, 0.5, 0.4]
        viability, series = _number(wells)
        assert len(series) == 1
        assert viability.loc[viability["concentration"] == FIVE[0], "intensity"].item() == pytest.approx(0.3)

    def test_a_single_repeated_dose_is_averaged_not_split(self) -> None:
        """CTRPv2 puts some compounds on a plate at one dose in 32 wells; that is no dose series."""
        _, series = _number(_wells(("c", "d", "P1", "e", [1.0] * 4, "x")))
        assert series[["series_key", "n_doses"]].values.tolist() == [["P1", 1]]

    def test_without_a_position_repeats_are_averaged(self) -> None:
        _, series = _number(_only_series_key(_wells(("c", "d", "P1", "e", FIVE + FIVE, "x"))))
        assert len(series) == 1


class TestShortSeries:
    def test_a_short_series_is_kept_in_the_table_but_not_fitted(self) -> None:
        wells = _wells(("c", "d", "P1", "e", FIVE, "x"), ("c", "d", "P2", "e", FIVE[:2], "x"))
        viability, series = _number(wells)
        assert series[["series_key", "eligible", "n_doses"]].values.tolist() == [["P1", True, 5], ["P2", False, 2]]
        assert set(viability["replicate"]) == {0}

    def test_eligible_series_are_numbered_first(self) -> None:
        wells = _wells(("c", "d", "P1", "e", FIVE[:2], "x"), ("c", "d", "P2", "e", FIVE, "x"))
        _, series = _number(wells)
        assert series[["series_key", "replicate"]].values.tolist() == [["P2", 0], ["P1", 1]]


class TestReplicateHierarchy:
    @staticmethod
    def _indices(series: pd.DataFrame) -> list[list]:
        return series[["series_key", "replicate", "biological_replicate", "technical_replicate"]].values.tolist()

    def test_series_sharing_an_experiment_are_technical_replicates_of_one_biological(self) -> None:
        wells = _wells(
            ("c", "d", "P1", "e1", FIVE, "x"),
            ("c", "d", "P2", "e1", FIVE, "x"),
            ("c", "d", "P3", "e2", FIVE, "x"),
        )
        _, series = _number(wells)
        assert self._indices(series) == [["P1", 0, 0, 0], ["P2", 1, 0, 1], ["P3", 2, 1, 0]]

    def test_replicates_are_numbered_by_experiment_before_key(self) -> None:
        wells = _wells(
            ("c", "d", "P1", "e2", FIVE, "x"),
            ("c", "d", "P2", "e1", FIVE, "x"),
            ("c", "d", "P3", "e2", FIVE, "x"),
        )
        viability, series = _number(wells)
        assert self._indices(series) == [["P2", 0, 0, 0], ["P1", 1, 1, 0], ["P3", 2, 1, 1]]
        assert sorted(viability["replicate"].unique()) == [0, 1, 2]

    def test_a_single_series_pair_is_the_first_of_each(self) -> None:
        _, series = _number(_wells(("c", "d", "P1", "e", FIVE, "x")))
        assert self._indices(series) == [["P1", 0, 0, 0]]

    def test_without_an_experiment_column_every_series_is_technical(self) -> None:
        wells = _wells(("c", "d", "P1", "e1", FIVE, "x"), ("c", "d", "P2", "e2", FIVE, "x"))
        _, series = _number(_only_series_key(wells))
        assert self._indices(series) == [["P1", 0, 0, 0], ["P2", 1, 0, 1]]

    def test_a_short_series_keeps_its_place_in_the_hierarchy_but_is_numbered_last(self) -> None:
        wells = _wells(
            ("c", "d", "P1", "e1", FIVE[:2], "x"),
            ("c", "d", "P2", "e1", FIVE, "x"),
            ("c", "d", "P3", "e2", FIVE, "x"),
        )
        viability, series = _number(wells)
        assert self._indices(series) == [["P2", 0, 0, 1], ["P3", 1, 1, 0], ["P1", 2, 0, 0]]
        assert sorted(viability["replicate"].unique()) == [0, 1]

    def test_a_split_repeat_block_is_two_technical_replicates(self) -> None:
        _, series = _number(_wells(("c", "d", "P1", "e", FIVE + FIVE, "x")))
        assert self._indices(series) == [["P1#0", 0, 0, 0], ["P1#1", 1, 0, 1]]

    def test_indices_restart_per_pair(self) -> None:
        wells = _wells(("c", "d", "P1", "e1", FIVE, "x"), ("c", "d2", "P1", "e2", FIVE, "x"))
        _, series = _number(wells)
        assert series[["biological_replicate", "technical_replicate"]].values.tolist() == [[0, 0], [0, 0]]


class TestMetadata:
    def test_metadata_names_are_lowercased(self) -> None:
        _, series = _number(_wells(("c", "d", "P1", "e", FIVE, "2020-01-01")))
        assert series.loc[0, "date"] == "2020-01-01"

    def test_several_values_in_one_series_are_joined(self) -> None:
        wells = _wells(("c", "d", "P1", "e", FIVE, "b"))
        wells.loc[0, "DATE"] = "a"
        _, series = _number(wells)
        assert series.loc[0, "date"] == "a;b"

    def test_a_missing_metadata_value_stays_missing(self) -> None:
        wells = _wells(("c", "d", "P1", "e", FIVE, "x"))
        wells["DATE"] = np.nan
        _, series = _number(wells)
        assert pd.isna(series.loc[0, "date"])


class TestLabels:
    def test_surrounding_whitespace_is_stripped(self) -> None:
        viability, _ = _number(_wells((" c ", "d", "P1", "e", FIVE, "x")))
        assert set(viability["sample"]) == {"c"}

    def test_a_pipe_in_a_label_is_rejected_and_named(self) -> None:
        with pytest.raises(InvalidInputError, match=r"'drug' contain '\|'.*'a\|b'"):
            _number(_wells(("c", "a|b", "P1", "e", FIVE, "x")))

    def test_a_missing_label_does_not_trip_the_guard(self) -> None:
        wells = _wells(("c", "d", "P1", "e", FIVE, "x"))
        wells.loc[0, "drug"] = None
        assert len(_number(wells).viability) == 4


class TestExperimentScoping:
    def test_the_same_replicate_label_in_two_experiments_is_two_series(self) -> None:
        _, series = _number(_wells(("c", "d", "P1", "e1", FIVE, "x"), ("c", "d", "P1", "e2", FIVE, "y")))
        assert series[["experiment", "series_key", "biological_replicate", "technical_replicate"]].values.tolist() == [
            ["e1", "P1", 0, 0],
            ["e2", "P1", 1, 0],
        ]
        assert series["date"].tolist() == ["x", "y"]
