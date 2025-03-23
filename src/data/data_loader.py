import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from typing import Dict, Any
from omegaconf import DictConfig


def load_data(cfg: DictConfig) -> Dict[str, Any]:
    """
    Load and split the breast cancer dataset.

    Args:
        cfg: Hydra configuration object

    Returns:
        Dictionary containing train and test data
    """
    # Fetch dataset
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=cfg.data.dataset_id)

    # Get features and target
    X = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets

    if "Diagnosis" in y.columns:
        y["Diagnosis"] = y["Diagnosis"].map({"M": 1, "B": 0})

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state,
        stratify=y,
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": X.columns.tolist(),
        "target_name": y.columns[0],
    }


if __name__ == "__main__":
    from hydra import initialize, compose

    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name="config")
        data_dict = load_data(cfg)
        print("Data loaded successfully!")
        print(f"Training set shape: {data_dict['X_train'].shape}")
        print(f"Test set shape: {data_dict['X_test'].shape}")
