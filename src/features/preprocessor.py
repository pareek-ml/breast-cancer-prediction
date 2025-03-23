import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from typing import Dict, Any
from omegaconf import DictConfig


def preprocess_data(cfg: DictConfig, data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess the data by scaling features and selecting the most important ones.

    Args:
        cfg: Hydra configuration object
        data_dict: Dictionary containing train and test data

    Returns:
        Dictionary containing preprocessed train and test data
    """
    # Initialize scaler
    scaler = StandardScaler()

    # Scale features
    X_train_scaled = scaler.fit_transform(data_dict["X_train"])
    X_test_scaled = scaler.transform(data_dict["X_test"])

    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=cfg.preprocessing.n_features)
    X_train_selected = selector.fit_transform(X_train_scaled, data_dict["y_train"])
    X_test_selected = selector.transform(X_test_scaled)

    # Get selected feature names
    selected_features = np.array(data_dict["feature_names"])[selector.get_support()]

    return {
        "X_train": X_train_selected,
        "X_test": X_test_selected,
        "y_train": data_dict["y_train"],
        "y_test": data_dict["y_test"],
        "feature_names": selected_features.tolist(),
        "target_name": data_dict["target_name"],
        "scaler": scaler,
        "selector": selector,
    }


if __name__ == "__main__":
    from hydra import initialize, compose

    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name="config")
        from data.data_loader import load_data

        data_dict = load_data(cfg)
        processed_dict = preprocess_data(cfg, data_dict)
        print("Data preprocessed successfully!")
        print(f"Selected features: {processed_dict['feature_names']}")
        print(
            f"Training set shape after preprocessing: {processed_dict['X_train'].shape}"
        )
        print(f"Test set shape after preprocessing: {processed_dict['X_test'].shape}")
