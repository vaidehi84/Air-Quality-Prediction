# Air Quality Prediction System

A professional Air Quality Prediction System built with Python, Scikit-learn and Tkinter. The project features a clean ML pipeline, data preprocessing, model evaluation, modern desktop GUI, and visualization support.

## Project Overview

This repository transforms the original notebook into a production-style portfolio project. It includes:
- End-to-end data preprocessing and training pipeline
- AQI category prediction using pollutant inputs
- Modern Tkinter desktop GUI with responsive controls
- Color-coded AQI category display, health insights, and advice
- Performance visuals saved to `screenshots/`

## Features

- Clean dataset loading and missing value handling
- Feature engineering, scaling, and encoded labels
- Random Forest classification with evaluation metrics
- Confusion matrix, classification report, and feature importance
- Modern GUI with prediction card, reset button, and health recommendations
- Auto-training fallback if model artifacts are missing

## Project Structure

```
Air-Quality-Prediction-System/
├── app.py
├── model/
│   ├── pipeline.py
│   ├── __init__.py
├── dataset/
│   └── global_air_pollution_dataset.csv
├── notebooks/
│   └── Air-Quality (1).ipynb
├── screenshots/
│   ├── category_distribution.png
│   ├── confusion_matrix.png
│   ├── correlation_matrix.png
│   ├── feature_importance.png
│   ├── gui_screenshot.png
│   └── prediction_comparison.png
├── assets/
│   └── README_PLACEHOLDER.txt
├── requirements.txt
├── README.md
├── .gitignore
└── utils/
    ├── data_utils.py
    └── __init__.py
```

## Installation

1. Clone this repository.
2. Create a virtual environment and activate it.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Application

```bash
python app.py
```

The application loads the sample dataset from `dataset/global_air_pollution_dataset.csv`. If a trained model artifact does not exist, it trains the model automatically and saves it under `model/`.

## GUI Usage

1. Enter numeric pollutant AQI values for CO, Ozone, NO2, and PM2.5.
2. Click **Predict AQI**.
3. View the predicted category, health recommendation, and color-coded result.
4. Use **Reset Inputs** to clear the form.

## Screenshots

The repository includes generated screenshots in `screenshots/`:
- `gui_screenshot.png`
- `prediction_comparison.png`
- `correlation_matrix.png`
- `feature_importance.png`
- `confusion_matrix.png`

These files are generated automatically during model training and can be used for README or portfolio display.

## Future Improvements

- Add a full dashboard with dynamic Matplotlib chart embedding
- Add model export support with ONNX or TensorFlow
- Add logging and configuration management
- Add a full Flask web version or PyQt desktop variant
