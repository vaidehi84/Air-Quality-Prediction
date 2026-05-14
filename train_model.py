from pathlib import Path

from utils.data_utils import load_dataset, prepare_training_data
from utils.model_utils import train_classifier, evaluate_model, save_model


def main():
    project_root = Path(__file__).resolve().parent
    dataset_path = project_root / "dataset" / "global_air_pollution_dataset.csv"
    model_path = project_root / "models" / "air_quality_model.pkl"

    print("Starting the Air Quality prediction model training pipeline...")
    try:
        df = load_dataset(dataset_path)
    except FileNotFoundError as error:
        print(f"Error: {error}")
        return

    X, y = prepare_training_data(df)
    model = train_classifier(X, y)
    report = evaluate_model(model, X, y)

    save_model(model, model_path)

    print("Model training completed successfully!")
    print(f"Model saved to: {model_path}")
    print(f"Accuracy: {report['accuracy'] * 100:.2f}%")
    print("Classification report:\n")
    print(report["report"])


if __name__ == "__main__":
    main()
