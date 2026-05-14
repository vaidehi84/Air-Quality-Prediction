from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils.data_utils import FEATURE_COLUMNS, TARGET_COLUMN, CATEGORY_LABELS, load_dataset, prepare_training_data


class AirQualityModel:
    def __init__(self, pipeline: Pipeline, encoder: LabelEncoder):
        self.pipeline = pipeline
        self.encoder = encoder

    def predict_category(self, features: pd.DataFrame) -> np.ndarray:
        encoded = self.pipeline.predict(features)
        return self.encoder.inverse_transform(encoded)


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])


def train_classifier(X: pd.DataFrame, y: pd.Series) -> AirQualityModel:
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    pipeline = build_pipeline()
    pipeline.fit(X, y_encoded)
    return AirQualityModel(pipeline, encoder)


def evaluate_model(model: AirQualityModel, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    encoded_predictions = model.pipeline.predict(X_test)
    predictions = model.encoder.inverse_transform(encoded_predictions)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "report": classification_report(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=CATEGORY_LABELS),
    }


def train_and_evaluate(dataset_path: Path, model_path: Path) -> Tuple[AirQualityModel, dict, pd.DataFrame]:
    df = load_dataset(dataset_path)
    X, y = prepare_training_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    model = train_classifier(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    metrics["cross_validation_score"] = cross_val_score(model.pipeline, X, LabelEncoder().fit_transform(y), cv=5).mean()
    save_model(model, model_path)
    return model, metrics, df


def save_model(model: AirQualityModel, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)


def load_model(source: Path) -> AirQualityModel:
    return joblib.load(source)
