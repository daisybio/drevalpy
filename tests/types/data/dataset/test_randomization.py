"""The randomization record ``Dataset.with_randomized_views`` writes, and the data it leaves alone."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.dataset._randomization import ViewRandomization
from drevalpy.types.enums import RandomizationType


class TestWithRandomizedViews:
    @pytest.mark.parametrize("randomization_type", list(RandomizationType))
    def test_the_views_type_and_seed_are_recorded(self, synthetic_dataset: Dataset, randomization_type):
        randomized = synthetic_dataset.with_randomized_views(
            ["methylation", "gene_expression"], randomization_type, random_state=7
        )

        assert randomized.view_randomization == ViewRandomization(
            frozenset({"gene_expression", "methylation"}), randomization_type, 7
        )
        provenance = randomized.randomization_provenance
        assert provenance is not None
        assert provenance == {
            "effective_seed": 7,
            "type": randomization_type.value,
            "views": ["gene_expression", "methylation"],
        }
        # A plain str, not the enum: anndata cannot write an enum into uns.
        assert type(provenance["type"]) is str

    def test_the_data_are_shared_and_left_alone(self, synthetic_dataset: Dataset):
        before = np.array(synthetic_dataset.mdata.mod["gene_expression"].X)

        randomized = synthetic_dataset.with_randomized_views(["gene_expression"], random_state=0)

        assert randomized.mdata.mod["gene_expression"] is synthetic_dataset.mdata.mod["gene_expression"]
        np.testing.assert_array_equal(randomized.mdata.mod["gene_expression"].X, before)
        assert randomized.mdata.uns["drug_graphs"] is synthetic_dataset.mdata.uns["drug_graphs"]

    def test_the_label_and_name_are_kept_in_the_copy_and_the_original_is_unrandomized(self, synthetic_dataset: Dataset):
        randomized = synthetic_dataset.with_randomized_views(
            ["gene_expression"], randomization=("SVRC", "gene_expression")
        )

        assert randomized.name == synthetic_dataset.name
        assert randomized.measure == synthetic_dataset.measure
        assert randomized.randomization == ("SVRC", "gene_expression")
        assert synthetic_dataset.randomization is None
        assert synthetic_dataset.view_randomization is None

    def test_without_a_seed_one_is_drawn_and_recorded(self, synthetic_dataset: Dataset):
        record = synthetic_dataset.with_randomized_views(["gene_expression"]).view_randomization

        assert record is not None
        assert 0 <= record.seed < 2**32

    def test_an_unknown_randomization_type_is_rejected(self, synthetic_dataset: Dataset):
        with pytest.raises(ValueError, match="not a valid RandomizationType"):
            synthetic_dataset.with_randomized_views(["gene_expression"], randomization_type="shuffle")  # type: ignore[arg-type]
