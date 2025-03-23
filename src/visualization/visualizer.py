import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any
from omegaconf import DictConfig
import os


def create_visualizations(cfg: DictConfig, data_dict: Dict[str, Any]) -> None:
    """
    Create various visualizations for data analysis.

    Args:
        cfg: Hydra configuration object
        data_dict: Dictionary containing train and test data
    """
    # Create save directory if it doesn't exist
    os.makedirs(cfg.visualization.save_path, exist_ok=True)

    # Set style
    plt.style.use(cfg.visualization.style)

    # 1. Feature Distribution Plot
    plt.figure(figsize=cfg.visualization.figsize)
    for i, feature in enumerate(
        data_dict["feature_names"][:5]
    ):  # Plot first 5 features
        plt.subplot(2, 3, i + 1)
        sns.histplot(
            data=data_dict["X_train"],
            x=feature,
            hue=data_dict["y_train"].values.ravel(),
        )
        plt.title(f"Distribution of {feature}")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.visualization.save_path, "feature_distribution.png"))
    plt.close()

    # 2. Correlation Matrix
    plt.figure(figsize=(12, 10))
    correlation_matrix = pd.DataFrame(
        data_dict["X_train"], columns=data_dict["feature_names"]
    ).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.visualization.save_path, "correlation_matrix.png"))
    plt.close()

    # 3. Target Distribution
    plt.figure(figsize=cfg.visualization.figsize)
    sns.countplot(
        data=pd.DataFrame(data_dict["y_train"], columns=[data_dict["target_name"]])
    )
    plt.title("Distribution of Target Variable")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.visualization.save_path, "target_distribution.png"))
    plt.close()


def plot_model_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_importance: np.ndarray = None,
    feature_names: list = None,
    save_path: str = None,
) -> None:
    """
    Create visualizations for model results.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        feature_importance: Feature importance scores (optional)
        feature_names: List of feature names (optional)
        save_path: Path to save the plots
    """
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = pd.crosstab(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
    plt.close()

    # 2. Feature Importance (if provided)
    if feature_importance is not None and feature_names is not None:
        plt.figure(figsize=(12, 6))
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": feature_importance}
        ).sort_values("importance", ascending=False)

        sns.barplot(data=importance_df, x="importance", y="feature")
        plt.title("Feature Importance")
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, "feature_importance.png"))
        plt.close()


if __name__ == "__main__":
    from hydra import initialize, compose

    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name="config")
        from data.data_loader import load_data

        data_dict = load_data(cfg)
        create_visualizations(cfg, data_dict)
        print("Visualizations created successfully!")
