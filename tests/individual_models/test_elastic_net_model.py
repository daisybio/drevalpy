import pytest
import tempfile
from sklearn.linear_model import Ridge, ElasticNet

from drevalpy.evaluation import evaluate
from drevalpy.models import ElasticNetModel
from .utils import sample_dataset


def test_train(sample_dataset):
    drug_response, cell_line_input, drug_input = sample_dataset
    for test_mode in ["LPO", "LCO", "LDO"]:
        drug_response.split_dataset(
            n_cv_splits=5,
            mode=test_mode,
        )
        hpams = ElasticNetModel.get_hyperparameter_set()
        e_net = ElasticNetModel()
        for hpam_combi in hpams:
            e_net.build_model(hpam_combi)
            if hpam_combi["l1_ratio"] == 0.0:
                assert issubclass(type(e_net.model), Ridge)
            else:
                assert issubclass(type(e_net.model), ElasticNet)

            split = drug_response.cv_splits[0]
            train_dataset = split["train"]
            val_dataset = split["validation"]
            e_net.train(
                output=train_dataset,
                cell_line_input=cell_line_input,
                drug_input=drug_input,
            )
            val_dataset.predictions = e_net.predict(
                drug_ids=val_dataset.drug_ids,
                cell_line_ids=val_dataset.cell_line_ids,
                drug_input=drug_input,
                cell_line_input=cell_line_input,
            )
            assert val_dataset.predictions is not None
            metrics = evaluate(val_dataset, metric=["Pearson"])
            assert metrics['Pearson'] > -0.5

        tmp = tempfile.NamedTemporaryFile()
        with pytest.raises(NotImplementedError):
            e_net.save(path=tmp.name)

        with pytest.raises(NotImplementedError):
            e_net.load(path=tmp.name)