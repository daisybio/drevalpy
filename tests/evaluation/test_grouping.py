"""Tests for ``drevalpy.evaluation._grouping``: the labels of the built-in groupings."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.evaluation._grouping import GROUPINGS, defined_labels, group_labels
from tests.synthetic import make_run_result


class TestGroupLabels:
    def test_drug_sample_and_study_groupings(self):
        run = make_run_result(n_pairs=4)
        run.sample_ids = np.array(["GDSC2::CL_1", "CTRPv2::CL_1", "GDSC2::a::b", "CTRPv2::CL_2"])

        assert GROUPINGS == ("drug", "sample", "study")
        np.testing.assert_array_equal(group_labels(run, "drug"), run.drug_ids)
        np.testing.assert_array_equal(group_labels(run, "sample"), run.sample_ids)
        assert group_labels(run, "study").tolist() == ["GDSC2", "CTRPv2", "GDSC2", "CTRPv2"]
        assert defined_labels(group_labels(run, "drug")).all()

    @pytest.mark.parametrize("sample_id", ["CL_1", "::CL_1"])
    def test_a_sample_id_without_a_study_prefix_has_no_study(self, sample_id):
        run = make_run_result(n_pairs=2)
        run.sample_ids = np.array(["GDSC2::CL_1", sample_id])

        labels = group_labels(run, "study")

        assert labels[0] == "GDSC2"
        assert labels[1] is np.nan
        assert defined_labels(labels).tolist() == [True, False]

    def test_an_unknown_grouping_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown grouping 'tissue'"):
            group_labels(make_run_result(), "tissue")
