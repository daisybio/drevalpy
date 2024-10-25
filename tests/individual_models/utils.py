import os
import tempfile
from typing import Tuple
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
import pytest
import requests
import zipfile
import pickle


def call_save_and_load(model):
    tmp = tempfile.NamedTemporaryFile()
    with pytest.raises(NotImplementedError):
        model.save(path=tmp.name)
    with pytest.raises(NotImplementedError):
        model.load(path=tmp.name)
