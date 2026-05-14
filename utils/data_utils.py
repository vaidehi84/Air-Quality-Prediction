import numpy as np
import pandas as pd
from pathlib import Path

FEATURE_COLUMNS = [
    "PM2.5 AQI Value",
    "PM10 AQI Value",
    "NO2 AQI Value",
    "SO2 AQI Value",
    "CO AQI Value",
    "Ozone AQI Value",
]
TARGET_COLUMN = "AQI Category"
AQI_VALUE_COLUMN = "AQI Value"
CATEGORY_THRESHOLDS = [50, 100, 150, 200]
CATEGORY_LABELS = ["Good", "Moderate", "Poor", "Very Poor", "Severe"]

AQI_STYLES = {
    "Good": {
        "color": "#219653",
        "background": "#e6f7e8",
        "message": "Clean air. Great for outdoor activities.",
    },
    "Moderate": {
        "color": "#f2c94c",
        "background": "#fff7dd",
        "message": "Acceptable air quality; sensitive groups should stay aware.",
    },
    "Poor": {
        "color": "#f2994a",
        "background": "#fff1e5",
        "message": "Limit outdoor exposure and consider indoor air cleaning.",
    },
    "Very Poor": {
        "color": "#eb5757",
        "background": "#ffecec",
        "message": "Avoid outdoor activity and keep windows closed.",
    },
    "Severe": {
        "color": "#9b51e0",
        "background": "#f0e6ff",
        "message": "Dangerous air quality. Stay indoors and limit exposure.",
    },
}


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    data = pd.read_csv(dataset_path)

    if "PM10 AQI Value" not in data.columns:
        data["PM10 AQI Value"] = (
            data["PM2.5 AQI Value"].fillna(0) * 1.5
            + data["NO2 AQI Value"].fillna(0) * 0.12
            + 5
        ).round(1)

    if "SO2 AQI Value" not in data.columns:
        data["SO2 AQI Value"] = (
            data["NO2 AQI Value"].fillna(0) * 0.45
            + data["CO AQI Value"].fillna(0) * 0.5
            + 2
        ).round(1)

    if TARGET_COLUMN not in data.columns and AQI_VALUE_COLUMN in data.columns:
        data[TARGET_COLUMN] = data[AQI_VALUE_COLUMN].apply(compute_category_from_aqi)

    data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(str)
    data = data.dropna(subset=FEATURE_COLUMNS)
    data[TARGET_COLUMN] = data[TARGET_COLUMN].replace("", "Unknown")
    return data


def get_sample_dataset() -> pd.DataFrame:
    sample = pd.DataFrame([
        {
            "PM2.5 AQI Value": 18.2,
            "PM10 AQI Value": 40.1,
            "NO2 AQI Value": 28.3,
            "SO2 AQI Value": 14.5,
            "CO AQI Value": 8.9,
            "Ozone AQI Value": 22.4,
            "AQI Value": 30,
            "AQI Category": "Good",
        },
        {
            "PM2.5 AQI Value": 45.0,
            "PM10 AQI Value": 72.4,
            "NO2 AQI Value": 50.9,
            "SO2 AQI Value": 32.5,
            "CO AQI Value": 18.2,
            "Ozone AQI Value": 68.0,
            "AQI Value": 78,
            "AQI Category": "Moderate",
        },
        {
            "PM2.5 AQI Value": 82.5,
            "PM10 AQI Value": 120.1,
            "NO2 AQI Value": 95.0,
            "SO2 AQI Value": 55.4,
            "CO AQI Value": 40.5,
            "Ozone AQI Value": 120.0,
            "AQI Value": 130,
            "AQI Category": "Poor",
        },
        {
            "PM2.5 AQI Value": 145.0,
            "PM10 AQI Value": 210.0,
            "NO2 AQI Value": 160.2,
            "SO2 AQI Value": 95.6,
            "CO AQI Value": 78.1,
            "Ozone AQI Value": 210.0,
            "AQI Value": 180,
            "AQI Category": "Very Poor",
        },
        {
            "PM2.5 AQI Value": 265.2,
            "PM10 AQI Value": 360.5,
            "NO2 AQI Value": 280.7,
            "SO2 AQI Value": 155.0,
            "CO AQI Value": 160.0,
            "Ozone AQI Value": 325.0,
            "AQI Value": 290,
            "AQI Category": "Severe",
        },
    ])
    return sample


def compute_category_from_aqi(aqi_value: float) -> str:
    if aqi_value <= CATEGORY_THRESHOLDS[0]:
        return CATEGORY_LABELS[0]
    if aqi_value <= CATEGORY_THRESHOLDS[1]:
        return CATEGORY_LABELS[1]
    if aqi_value <= CATEGORY_THRESHOLDS[2]:
        return CATEGORY_LABELS[2]
    if aqi_value <= CATEGORY_THRESHOLDS[3]:
        return CATEGORY_LABELS[3]
    return CATEGORY_LABELS[4]


def get_health_recommendation(category: str) -> str:
    return AQI_STYLES.get(category, AQI_STYLES["Moderate"])["message"]


def prepare_training_data(df: pd.DataFrame):
    df = df.copy()
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].mean()).clip(lower=0)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y
