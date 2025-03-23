import hydra
from omegaconf import DictConfig
import os
from data.data_loader import load_data
from features.preprocessor import preprocess_data
from visualization.visualizer import create_visualizations, plot_model_results
from models.trainer import train_model


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function to run the complete pipeline.

    Args:
        cfg: Hydra configuration object
    """
    # Create necessary directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)

    # Load data
    print("Loading data...")
    data_dict = load_data(cfg)

    # Create initial visualizations
    print("Creating data visualizations...")
    create_visualizations(cfg, data_dict)

    # Preprocess data
    print("Preprocessing data...")
    processed_dict = preprocess_data(cfg, data_dict)

    # Train model
    print("Training model...")
    results = train_model(cfg, processed_dict)

    # Create model result visualizations
    print("Creating model result visualizations...")
    plot_model_results(
        results["y_true"],
        results["y_pred"],
        results["feature_importance"],
        processed_dict["feature_names"],
        "results/figures",
    )

    # Save metrics
    print("\nFinal Model Performance:")
    for metric, value in results["metrics"].items():
        print(f"{metric.capitalize()}: {value:.4f}")

    print("\nBest Parameters:")
    for param, value in results["best_params"].items():
        print(f"{param}: {value}")


if __name__ == "__main__":
    main()
