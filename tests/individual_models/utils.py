import tempfile

import pytest


def call_save_and_load(model):
    tmp = tempfile.NamedTemporaryFile()
    with pytest.raises(NotImplementedError):
        model.save(path=tmp.name)
    with pytest.raises(NotImplementedError):
        model.load(path=tmp.name)
