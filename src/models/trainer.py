import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from typing import Dict, Any
from omegaconf import DictConfig
import optuna
import joblib
import os


def objective(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray) -> float:
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial: Optuna trial object
        X_train: Training features
        y_train: Training labels

    Returns:
        Mean cross-validation score
    """
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
    }

    model = RandomForestClassifier(**param, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    return scores.mean()


def train_model(cfg: DictConfig, processed_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train and evaluate the model with hyperparameter optimization.

    Args:
        cfg: Hydra configuration object
        processed_dict: Dictionary containing preprocessed data

    Returns:
        Dictionary containing model results and metrics
    """
    # Create results directory
    os.makedirs("results/models", exist_ok=True)

    # Hyperparameter optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(
            trial, processed_dict["X_train"], processed_dict["y_train"].values.ravel()
        ),
        n_trials=50,
    )

    # Train final model with best parameters
    best_model = RandomForestClassifier(**study.best_params, random_state=42)
    best_model.fit(processed_dict["X_train"], processed_dict["y_train"].values.ravel())

    # Make predictions
    y_pred = best_model.predict(processed_dict["X_test"])
    y_true = processed_dict["y_test"].values.ravel()

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

    bundle = {
        "model": best_model,
        "scaler": processed_dict["scaler"],
        "selector": processed_dict["selector"],
        "feature_names": processed_dict["feature_names"],
    }

    # Save model
    joblib.dump(bundle, "results/models/best_model.joblib")

    # Get feature importance
    feature_importance = best_model.feature_importances_
    # print("Feature names: ", processed_dict["feature_names"])

    return {
        "model": best_model,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "best_params": study.best_params,
        "y_pred": y_pred,
        "y_true": y_true,
    }


if __name__ == "__main__":
    from hydra import initialize, compose

    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(config_name="config")
        from data.data_loader import load_data
        from features.preprocessor import preprocess_data

        # Load and preprocess data
        data_dict = load_data(cfg)
        processed_dict = preprocess_data(cfg, data_dict)

        # Train model
        results = train_model(cfg, processed_dict)

        # Print results
        print("\nModel Performance:")
        for metric, value in results["metrics"].items():
            print(f"{metric.capitalize()}: {value:.4f}")

        print("\nBest Parameters:")
        for param, value in results["best_params"].items():
            print(f"{param}: {value}")
