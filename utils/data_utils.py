import pandas as pd

FEATURE_COLUMNS = [
    "CO AQI Value",
    "Ozone AQI Value",
    "NO2 AQI Value",
    "PM2.5 AQI Value",
]
TARGET_COLUMN = "AQI Category"

CATEGORY_THRESHOLDS = [50, 100, 150, 200]
CATEGORY_LABELS = ["Good", "Moderate", "Poor", "Very Poor", "Severe"]

AQI_STYLES = {
    "Good": {
        "color": "#219653",
        "background": "#e6f7e8",
        "message": "Clean air. Great for outdoor activities!",
    },
    "Moderate": {
        "color": "#f2c94c",
        "background": "#fff7dd",
        "message": "Air is acceptable; sensitive groups should take caution.",
    },
    "Poor": {
        "color": "#f2994a",
        "background": "#fff1e5",
        "message": "Sensitive individuals should reduce prolonged outdoor exertion.",
    },
    "Very Poor": {
        "color": "#eb5757",
        "background": "#ffecec",
        "message": "Avoid outdoor activity and consider using air purification indoors.",
    },
    "Severe": {
        "color": "#9b51e0",
        "background": "#f0e6ff",
        "message": "Dangerous air quality. Stay indoors and limit exposure.",
    },
}


def load_dataset(dataset_path: str) -> pd.DataFrame:
    data = pd.read_csv(dataset_path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must contain '{TARGET_COLUMN}' column.")

    data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(str)
    data = data.dropna(subset=FEATURE_COLUMNS)
    data[TARGET_COLUMN] = data[TARGET_COLUMN].replace("", "Unknown")
    return data


def compute_aqi_category(aqi_value: float) -> str:
    if aqi_value <= CATEGORY_THRESHOLDS[0]:
        return CATEGORY_LABELS[0]
    if aqi_value <= CATEGORY_THRESHOLDS[1]:
        return CATEGORY_LABELS[1]
    if aqi_value <= CATEGORY_THRESHOLDS[2]:
        return CATEGORY_LABELS[2]
    if aqi_value <= CATEGORY_THRESHOLDS[3]:
        return CATEGORY_LABELS[3]
    return CATEGORY_LABELS[4]


def prepare_training_data(df: pd.DataFrame):
    df = df.copy()
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].mean())
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


from sklearn.model_selection import train_test_split


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X, y = prepare_training_data(df)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
