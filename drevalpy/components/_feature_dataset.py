"""Feature dataset container used internally by featurizers and _ComponentStack."""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from ..datasets.utils import permute_features, randomize_graph


class FeatureDataset:
    """Class for feature datasets.

    This class represents datasets with one or more views of features associated with a set of entities,
    such as drugs or cell lines. The feature data is stored in a nested dictionary structure::

        {
            identifier_1: {
                view_name_1: feature_vector,
                view_name_2: feature_vector,
                ...
            },
            identifier_2: {
                view_name_1: feature_vector,
                view_name_2: feature_vector,
                ...
            },
            ...
        }

    - Each outer key is a string identifier (e.g. a cell line ID or drug ID)
    - Each inner key is the name of a view (e.g. 'gene_expression', 'fingerprints')
    - Each inner value is a feature vector or object representing that view for the identifier
    """

    _features: dict[str, dict[str, Any]]
    _meta_info: dict[str, Any]

    @classmethod
    def from_csv(
        cls: type[FeatureDataset],
        path_to_csv: str | Path,
        id_column: str,
        view_name: str,
        drop_columns: list[str] | None = None,
        transpose: bool = False,
        extract_meta_info: bool = True,
    ):
        """Load a one-view feature dataset from a csv file.

        Load a feature dataset from a csv file. The rows of the csv file represent the instances (cell lines or drugs),
        the columns represent the features. A column named id_column contains the identifiers of the instances.
        All unrelated columns (e.g. other id columns) should be provided as drop_columns,
        that will be removed from the dataset.

        :param path_to_csv: path to the csv file containing the data to be loaded
        :param view_name: name of the view (e.g. gene_expression)
        :param id_column: name of the column containing the identifiers
        :param drop_columns: list of columns to drop (e.g. other identifier columns)
        :param transpose: if True, the csv is transposed, i.e. the rows become columns and vice versa
        :param extract_meta_info: If ``True``, extract view metadata such as gene names.
        :returns: ``FeatureDataset`` loaded from the CSV file.
        """
        data = pd.read_csv(path_to_csv).T if transpose else pd.read_csv(path_to_csv)
        data[id_column] = data[id_column].astype(str)
        ids = data[id_column].values
        data_features = data.drop(columns=(drop_columns or []))
        data_features = data_features.set_index(id_column)
        data_features = data_features[~data_features.index.duplicated(keep="first")]
        features = {}

        for identifier in ids:
            features_for_instance = data_features.loc[identifier].values
            features[identifier] = {view_name: features_for_instance}

        meta_info = {}
        if extract_meta_info:
            meta_info = {view_name: list(data_features.columns)}

        return cls(features=features, meta_info=meta_info)

    def to_csv(self, path: str | Path, id_column: str, view_name: str):
        """Save the feature dataset to a CSV file.

        When meta_info is available for the view and valid, column names are taken
        from the metadata.

        :param path: Path to the CSV file.
        :param id_column: Name of the column containing the identifiers.
        :param view_name: Name of the view.
        """
        from ..datasets.feature_dataset_csv import feature_dataset_to_csv

        feature_dataset_to_csv(self, path, id_column, view_name)

    @property
    def meta_info(self) -> dict[str, Any]:
        """Returns the meta information.

        :returns: Meta information of this FeatureDataset
        """
        return self._meta_info

    @property
    def features(self) -> dict[str, dict[str, Any]]:
        """Returns the features.

        :returns: features of this FeatureDataset
        """
        return self._features

    @property
    def identifiers(self) -> np.ndarray:
        """Returns the identifiers of the features.

        Used in the pipeline.

        :returns: feature identifiers of this FeatureDataset
        """
        return np.array(list(self.features.keys()))

    @property
    def view_names(self) -> list[str]:
        """Returns the view_names.

        :returns: view_names of this FeatureDataset
        """
        return list(self.features[list(self.features.keys())[0]].keys())

    def __init__(
        self,
        features: dict[str, dict[str, Any]],
        meta_info: dict[str, Any] | None = None,
    ):
        """Initializes the feature dataset.

        :param features: Nested mapping ``{entity_id: {view_name: feature_vector}}``.
        :param meta_info: Optional per-view metadata, for example gene names.
        :raises AssertionError: if meta_info keys are not in view names
        """
        super().__init__()
        self._features = features
        self._meta_info = meta_info if meta_info is not None else {}
        if meta_info is not None:
            if not all(meta_key in self.view_names for meta_key in meta_info.keys()):
                raise AssertionError(f"Meta keys {meta_info.keys()} not in view names {self.view_names}")
            self._meta_info = meta_info

    def randomize_features(self, views_to_randomize: str | list[str], randomization_type: str) -> None:
        """Randomizes the feature vectors.

        Permutation permutes the feature vectors.
        Invariant means that the randomization is done in a way that a key characteristic of the feature is
        preserved. In case of matrices, this is the mean and standard deviation of the feature view for this
        instance, for networks it is the degree distribution.

        :param views_to_randomize: Feature view name or list of view names to randomize.
        :param randomization_type: ``permutation`` or ``invariant`` randomization mode.
        :raises AssertionError: if randomization_type is not 'permutation' or 'invariant'
        :raises ValueError: if no invariant randomization is available for the feature view type
        """
        if randomization_type not in ["permutation", "invariant"]:
            raise AssertionError(
                f"Unknown randomization type {randomization_type!r}. Choose from 'permutation', 'invariant'."
            )

        if isinstance(views_to_randomize, str):
            views_to_randomize = [views_to_randomize]

        if randomization_type == "permutation":
            self._features = permute_features(
                features=self.features,
                views_to_permute=views_to_randomize,
                identifiers=self.identifiers,
                all_views=self.view_names,
            )

        elif randomization_type == "invariant":
            for view in views_to_randomize:
                for identifier in self.identifiers:
                    if isinstance(self.features[identifier][view], np.ndarray):
                        new_features = np.random.normal(
                            self.features[identifier][view].mean(),
                            self.features[identifier][view].std(),
                            self.features[identifier][view].shape,
                        )
                    elif isinstance(self.features[identifier][view], nx.classes.graph.Graph):
                        new_features = randomize_graph(self.features[identifier][view])

                    else:
                        raise ValueError(
                            f"No invariant randomization available for feature view "
                            f"type {type(self.features[identifier][view])!r}."
                        )
                    self.features[identifier][view] = new_features

    def get_feature_matrix(self, view: str, identifiers: np.ndarray) -> np.ndarray:
        """Returns the feature matrix for the given view.

        The feature view must be a vector or matrix.

        :param view: view name
        :param identifiers: list of identifiers (cell lines oder drugs)
        :returns: feature matrix
        :raises AssertionError: if no identifiers are given
        :raises AssertionError: if view is not in the FeatureDataset
        :raises AssertionError: if identifiers are not in the FeatureDataset
        :raises AssertionError: if feature vectors of view have different lengths
        :raises AssertionError: if view is not a numpy array, i.e. not a vector or matrix
        """
        if len(identifiers) == 0:
            raise AssertionError("get_feature_matrix: No identifiers given.")

        if view not in self.view_names:
            raise AssertionError(f"View {view!r} not in in the FeatureDataset.")
        missing_identifiers = {id_ for id_ in identifiers if id_ not in self.identifiers}
        if missing_identifiers:
            raise AssertionError(
                f"{len(missing_identifiers)} of {len(np.unique(identifiers))} ids are not in the "
                f"FeatureDataset. Missing ids: {missing_identifiers}"
            )

        if not all(len(self.features[id_][view]) == len(self.features[identifiers[0]][view]) for id_ in identifiers):
            raise AssertionError(f"Feature vectors of view {view} have different lengths.")

        if not all(isinstance(self.features[id_][view], np.ndarray) for id_ in identifiers):
            raise AssertionError(f"get_feature_matrix only works for vectors or matrices. {view} is not a numpy array.")
        out = np.array([self.features[id_][view] for id_ in identifiers])
        return out

    def copy(self):
        """Returns a copy of the feature dataset.

        :returns: copy of the dataset
        """
        return FeatureDataset(features=copy.deepcopy(self.features), meta_info=copy.deepcopy(self.meta_info))

    def add_features(self, other: FeatureDataset) -> None:
        """Adds features views from another dataset. Inner join (only common identifiers are kept).

        :param other: other dataset
        :raises AssertionError: if feature views overlap
        """
        if len(set(self.view_names) & set(other.view_names)) != 0:
            raise AssertionError(
                "Trying to add features but feature views overlap. FeatureDatasets should be distinct."
            )
        if other.meta_info:
            self.add_meta_info(other)

        common_identifiers = set(self.identifiers).intersection(other.identifiers)
        new_features = {}
        for id_ in common_identifiers:
            id_ = str(id_)
            new_features[id_] = {view: self.features[id_][view] for view in self.view_names}
            for view in other.view_names:
                new_features[id_][view] = other.features[id_][view]

        self._features = new_features

    def add_meta_info(self, other: FeatureDataset) -> None:
        """Adds meta information to the feature dataset.

        :param other: other dataset
        """
        other_meta = other.meta_info
        if self.meta_info is None:
            self.meta_info = other_meta
        else:
            if other_meta is not None:
                self.meta_info.update(other_meta)

    def transform_features(self, ids: np.ndarray, transformer: TransformerMixin, view: str):
        """Applies a transformation like standard scaling to features.

        :param ids: The IDs to transform
        :param transformer: fitted sklearn transformer
        :param view: the view to transform
        :raises AssertionError: if view is not in the FeatureDataset
        :raises AssertionError: if a cell line is missing
        :raises AssertionError: if IDs are not unique
        """
        if view not in self.view_names:
            raise AssertionError(f"Transform view {view!r} not in in the FeatureDataset.")
        if not all([clid in self.features for clid in ids]):
            raise AssertionError("Trying to transform, but a cell line is missing.")

        if len(np.unique(ids)) != len(ids):
            raise AssertionError("IDs should be unique.")

        for identifier in ids:
            feature_vector = self.features[identifier][view]
            scaled_feature_vector = transformer.transform([feature_vector])[0]
            self.features[identifier][view] = scaled_feature_vector

    def fit_transform_features(self, train_ids: np.ndarray, transformer: TransformerMixin, view: str):
        """Fits and applies a transformation. Fitting is done only on the train_ids.

        :param train_ids: The IDs corresponding to the training dataset.
        :param transformer: sklearn transformer
        :param view: the view to transform
        :returns: The modified FeatureDataset with transformed gene expression features.
        :raises AssertionError: if view is not in the FeatureDataset
        :raises AssertionError: if train IDs are not unique
        """
        if view not in self.view_names:
            raise AssertionError(f"Transform view {view!r} not in in the FeatureDataset.")

        if len(np.unique(train_ids)) != len(train_ids):
            print(f"Train IDs: {train_ids}")

            raise AssertionError("Train IDs should be unique.")

        train_features = np.vstack([self.features[identifier][view] for identifier in train_ids])
        transformer.fit(train_features)

        for identifier in self.features:
            feature_vector = self.features[identifier][view]
            transformed_vector = transformer.transform([feature_vector])[0]
            self.features[identifier][view] = transformed_vector
        return transformer

    def apply(self, function: Callable, view: str):
        """Applies a function to the features of a view.

        :param function: function to apply
        :param view: view to apply the function to
        """
        for identifier in self.features:
            self.features[identifier][view] = function(self.features[identifier][view])
