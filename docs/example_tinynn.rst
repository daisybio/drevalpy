Example: TinyNN (Neural Network with PyTorch)
---------------------------------------------

In this example, we implement a simple feedforward neural network for drug response prediction using gene expression and drug fingerprint features.
We use and recommend PyTorch, but you can use any other framework like TensorFlow, JAX, etc.
Gene expression features are standardized using a ``StandardScaler``, while fingerprint features are used as-is.

1. We define a minimal PyTorch model with CPU/GPU support.

.. code-block:: Python

    import torch
    import torch.nn as nn
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class FeedForwardNetwork(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.to(device)

        def fit(self, x: np.ndarray, y: np.ndarray, lr: float = 1e-3, epochs: int = 100):
            self.train()
            x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
            y_tensor = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)

            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            loss_fn = nn.MSELoss()

            for _ in range(epochs):
                optimizer.zero_grad()
                loss = loss_fn(self(x_tensor), y_tensor)
                loss.backward()
                optimizer.step()

        def forward(self, x):
            return self.net(x)

        def predict(self, x: np.ndarray) -> np.ndarray:
            self.eval()
            with torch.no_grad():
                x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
                preds = self(x_tensor).squeeze(1)
                return preds.cpu().numpy()

2. We create the ``TinyNN`` model class that inherits from ``DRPModel``.

.. code-block:: Python

    from drevalpy.models.drp_model import DRPModel
    from drevalpy.datasets.dataset import FeatureDataset
    from sklearn.preprocessing import StandardScaler
    from drevalpy.models.utils import load_and_select_gene_features, load_drug_fingerprint_features


    class TinyNN(DRPModel):
        cell_line_views = ["gene_expression"]
        drug_views = ["fingerprints"]
        early_stopping = True

        def __init__(self):
            super().__init__()
            self.model = None
            self.hyperparameters = None
            self.scaler_gex = StandardScaler()

        @classmethod
        def get_model_name(cls) -> str:
            return "TinyNN"

3. We define how the features are loaded. Here, we use our presupplied datasets and the preimplemented functions. Loading features can be customized (Have a look at the FeatureDataset class for more details e.g. on how to load features from a CSV).

.. code-block:: Python

        def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:

            return load_and_select_gene_features(feature_type="gene_expression",
                                                data_path=data_path,
                                                dataset_name=dataset_name,
                                                gene_list="landmark_genes_reduced")


        def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:

            return load_drug_fingerprint_features(data_path, dataset_name, fill_na=True)

1. In the ``build_model`` we just store the hyperparameters.

.. code-block:: Python

        def build_model(self, hyperparameters: dict[str, Any]) -> None:
            self.hyperparameters = hyperparameters

5. In the train method we scale gene expression and train the model.

.. code-block:: Python

        def train(self, output, cell_line_input, drug_input, output_earlystopping=None, model_checkpoint_dir=None):
            gex = cell_line_input.get_feature_matrix("gene_expression", output.cell_line_ids)
            fp = drug_input.get_feature_matrix("fingerprints", output.drug_ids)

            gex = self.scaler_gex.fit_transform(gex)
            x = np.concatenate([gex, fp], axis=1)
            y = output.response

            self.model = FeedForwardNetwork(
                input_dim=x.shape[1],
                hidden_dim=self.hyperparameters["hidden_dim"]
            )
            self.model.fit(x, y)

6. We apply scaling in ``predict`` and return model outputs.

.. code-block:: Python

        def predict(self, cell_line_ids, drug_ids, cell_line_input, drug_input):
            gex = cell_line_input.get_feature_matrix("gene_expression", cell_line_ids)
            fp = drug_input.get_feature_matrix("fingerprints", drug_ids)

            gex = self.scaler_gex.transform(gex)
            x = np.concatenate([gex, fp], axis=1)

            return self.model.predict(x)

7. Add hyperparameters to your ``hyperparameters.yaml``. We add two values for the hidden layer size. DrEval will tune over this hyperparameter space.

.. code-block:: YAML

    TinyNN:
      hidden_dim:
        - 32
        - 64

8. Register the model in ``models/__init__.py``.

.. code-block:: Python

    from .your_model_folder.tinynn import TinyNN
    MULTI_DRUG_MODEL_FACTORY.update({"TinyNN": TinyNN})



