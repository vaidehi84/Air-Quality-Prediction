import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils.data_utils import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset, CATEGORY_LABELS


MODEL_FILENAME = Path("model") / "air_quality_model.pkl"


class ModelPipeline:
    def __init__(self, pipeline, encoder):
        self.pipeline = pipeline
        self.encoder = encoder

    def predict_category(self, features):
        encoded_prediction = self.pipeline.predict(features)
        return self.encoder.inverse_transform(encoded_prediction)


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> ModelPipeline:
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_train)
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_encoded)
    return ModelPipeline(pipeline, encoder)


def evaluate_model(model: ModelPipeline, X_test: pd.DataFrame, y_test: pd.Series):
    encoded_predictions = model.pipeline.predict(X_test)
    y_pred = model.encoder.inverse_transform(encoded_predictions)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=CATEGORY_LABELS)
    return {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm,
        "predictions": y_pred,
    }


def save_model(model: ModelPipeline, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as file:
        pickle.dump({"pipeline": model.pipeline, "encoder": model.encoder}, file)


def load_model(source: Path) -> ModelPipeline:
    with open(source, "rb") as file:
        artifact = pickle.load(file)
    return ModelPipeline(artifact["pipeline"], artifact["encoder"])


def generate_visualizations(
    df: pd.DataFrame,
    model: ModelPipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    report_data: dict,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    target_names = CATEGORY_LABELS

    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[FEATURE_COLUMNS].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_matrix.png")
    plt.close()

    # Feature importance
    importance = model.pipeline.named_steps["classifier"].feature_importances_
    plt.figure(figsize=(8, 5))
    plt.barh(FEATURE_COLUMNS, importance, color=["#3f72af", "#dbe2ef", "#bfc0c0", "#112d4e"])
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png")
    plt.close()

    # Category distribution
    category_counts = df[TARGET_COLUMN].value_counts().reindex(target_names).fillna(0)
    plt.figure(figsize=(8, 4))
    plt.bar(category_counts.index, category_counts.values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
    plt.title("AQI Category Distribution")
    plt.xlabel("AQI Category")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "category_distribution.png")
    plt.close()

    # Confusion matrix
    cm = report_data["confusion_matrix"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.title("Prediction Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()


def train_and_evaluate(dataset_path: Path, model_path: Path, screenshot_dir: Path):
    df = load_dataset(dataset_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    save_model(model, model_path)
    generate_visualizations(df, model, X_test, y_test, metrics, screenshot_dir)

    cross_val = cross_val_score(model.pipeline, X, LabelEncoder().fit_transform(y), cv=5)
    metrics["cross_validation_score"] = cross_val.mean()
    return {
        "model": model,
        "metrics": metrics,
        "dataset": df,
    }
