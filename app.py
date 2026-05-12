import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

import pandas as pd
from model.pipeline import load_model, train_and_evaluate
from utils.data_utils import AQI_STYLES, FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "dataset" / "global_air_pollution_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "air_quality_model.pkl"
SCREENSHOT_DIR = PROJECT_ROOT / "screenshots"


def load_or_train_model():
    if MODEL_PATH.exists():
        try:
            return load_model(MODEL_PATH), None
        except Exception as error:
            print("Model load failed:", error)
    result = train_and_evaluate(DATA_PATH, MODEL_PATH, SCREENSHOT_DIR)
    return result["model"], result["metrics"]


def create_main_window(model, performance_metrics):
    root = tk.Tk()
    root.title("Air Quality Prediction System")
    root.geometry("900x620")
    root.resizable(False, False)
    root.configure(bg="#eef3f7")

    style = ttk.Style(root)
    style.configure("Header.TLabel", background="#eef3f7", foreground="#1f3566", font=("Segoe UI", 20, "bold"))
    style.configure("Subtitle.TLabel", background="#eef3f7", foreground="#4f5b7d", font=("Segoe UI", 11))
    style.configure("Section.TLabelframe", background="#eef3f7", borderwidth=0)
    style.configure("Section.TLabelframe.Label", font=("Segoe UI", 14, "bold"), foreground="#1a2f52")

    header_frame = tk.Frame(root, bg="#eef3f7")
    header_frame.pack(fill="x", padx=20, pady=(20, 8))

    title_label = ttk.Label(header_frame, text="Air Quality Prediction Dashboard", style="Header.TLabel")
    title_label.pack(anchor="w")

    subtitle_label = ttk.Label(header_frame, text="Enter pollutant AQI values and get an instant category prediction with safety advice.", style="Subtitle.TLabel")
    subtitle_label.pack(anchor="w", pady=(6, 0))

    content_frame = tk.Frame(root, bg="#eef3f7")
    content_frame.pack(fill="both", expand=True, padx=20, pady=10)

    input_frame = tk.Frame(content_frame, bg="#ffffff", bd=0, relief=tk.FLAT)
    input_frame.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=4)
    input_frame.configure(highlightbackground="#d3dae6", highlightthickness=1, padx=20, pady=20)

    input_title = tk.Label(input_frame, text="Pollutant Inputs", font=("Segoe UI", 14, "bold"), bg="#ffffff", fg="#26334d")
    input_title.pack(anchor="w", pady=(0, 12))

    field_frame = tk.Frame(input_frame, bg="#ffffff")
    field_frame.pack(fill="x")

    entries = {}
    hints = [
        "CO AQI Value (0-200)",
        "Ozone AQI Value (0-300)",
        "NO2 AQI Value (0-200)",
        "PM2.5 AQI Value (0-500)",
    ]

    for index, label_text in enumerate(hints):
        label = tk.Label(field_frame, text=label_text, font=("Segoe UI", 11), bg="#ffffff", fg="#455a75")
        label.grid(row=index, column=0, sticky="w", pady=10)
        entry = tk.Entry(field_frame, font=("Segoe UI", 11), width=18, bd=1, relief=tk.SOLID)
        entry.grid(row=index, column=1, sticky="e", pady=10, padx=(12, 0))
        entries[label_text] = entry

    button_frame = tk.Frame(input_frame, bg="#ffffff")
    button_frame.pack(fill="x", pady=(14, 0))

    predict_button = tk.Button(button_frame, text="Predict AQI", command=lambda: predict_aqi(entries, result_widgets, model, performance_metrics),
                               bg="#2e6ef7", fg="white", activebackground="#254fe2", font=("Segoe UI", 11, "bold"), bd=0, padx=18, pady=10)
    predict_button.pack(side="left", padx=(0, 8))

    reset_button = tk.Button(button_frame, text="Reset Inputs", command=lambda: reset_fields(entries, result_widgets),
                             bg="#f0f4ff", fg="#2e6ef7", activebackground="#dbe4ff", font=("Segoe UI", 11, "bold"), bd=0, padx=18, pady=10)
    reset_button.pack(side="left")

    result_frame = tk.Frame(content_frame, bg="#ffffff", bd=0, relief=tk.FLAT)
    result_frame.pack(side="right", fill="both", expand=True, padx=(10, 0), pady=4)
    result_frame.configure(highlightbackground="#d3dae6", highlightthickness=1, padx=20, pady=20)

    result_title = tk.Label(result_frame, text="Prediction Summary", font=("Segoe UI", 14, "bold"), bg="#ffffff", fg="#26334d")
    result_title.pack(anchor="w", pady=(0, 12))

    card_frame = tk.Frame(result_frame, bg="#f8fbff", bd=0, relief=tk.RIDGE)
    card_frame.pack(fill="both", expand=True, padx=4, pady=4)
    card_frame.configure(padx=18, pady=18)

    category_label = tk.Label(card_frame, text="Category: —", font=("Segoe UI", 16, "bold"), bg="#f8fbff", fg="#2e6fce")
    category_label.pack(anchor="w", pady=(0, 12))

    score_label = tk.Label(card_frame, text="Estimated average AQI score: —", font=("Segoe UI", 11), bg="#f8fbff", fg="#5d6c83")
    score_label.pack(anchor="w", pady=(0, 6))

    advice_label = tk.Label(card_frame, text="Health recommendation will appear here.", wraplength=300, justify="left",
                            font=("Segoe UI", 11), bg="#f8fbff", fg="#4d586d")
    advice_label.pack(anchor="w", pady=(0, 12))

    status_label = tk.Label(card_frame, text="Application ready to predict air quality.", font=("Segoe UI", 10), bg="#f8fbff", fg="#627099")
    status_label.pack(anchor="w")

    metrics_frame = tk.Frame(result_frame, bg="#ffffff")
    metrics_frame.pack(fill="x", pady=(16, 0))

    throughput_label = tk.Label(metrics_frame, text="Model accuracy and training details", font=("Segoe UI", 12, "bold"), bg="#ffffff", fg="#26334d")
    throughput_label.pack(anchor="w")

    metrics_text = tk.StringVar()
    metrics_text.set(get_metrics_summary(performance_metrics))
    metrics_label = tk.Label(metrics_frame, textvariable=metrics_text, wraplength=320, justify="left",
                             font=("Segoe UI", 10), bg="#ffffff", fg="#4b5d78")
    metrics_label.pack(anchor="w", pady=(6, 0))

    return root, entries, {
        "category_label": category_label,
        "score_label": score_label,
        "advice_label": advice_label,
        "status_label": status_label,
        "metrics_text": metrics_text,
    }


def get_metrics_summary(metrics):
    if not metrics:
        return "Model artifacts loaded from disk. Run the app to verify prediction behavior."
    return (
        f"Accuracy: {metrics['accuracy'] * 100:.1f}% | "
        f"Cross-validation score: {metrics.get('cross_validation_score', 0) * 100:.1f}%"
    )


def format_prediction_result(category: str, average_score: float):
    style = AQI_STYLES.get(category, AQI_STYLES["Moderate"])
    return style, f"Category: {category}", f"Estimated average AQI score: {average_score:.1f}", style["message"]


def predict_aqi(entries, widgets, model, metrics):
    try:
        values = [float(entry.get()) for entry in entries.values()]
        if any(value < 0 for value in values):
            raise ValueError("Values must be zero or positive.")

        feature_vector = pd.DataFrame([{col: values[index] for index, col in enumerate(FEATURE_COLUMNS)}])
        prediction = model.predict_category(feature_vector)[0]
        average_value = sum(values) / len(values)

        style, category_text, score_text, advice_text = format_prediction_result(prediction, average_value)
        widgets["category_label"].config(text=category_text, fg=style["color"], bg=style["background"])
        widgets["score_label"].config(text=score_text, bg=style["background"])
        widgets["advice_label"].config(text=advice_text, bg=style["background"])
        widgets["status_label"].config(text="Prediction complete. Stay informed and stay safe.", bg=style["background"])
        for key, widget in widgets.items():
            if key == "metrics_text":
                continue
            widget.config(bg=style["background"])
        if metrics:
            widgets["metrics_text"].set(get_metrics_summary(metrics))

    except ValueError as error:
        messagebox.showerror("Invalid input", f"Please enter valid numeric pollutant values. {error}")


def reset_fields(entries, widgets):
    for entry in entries.values():
        entry.delete(0, tk.END)
    widgets["category_label"].config(text="Category: —", fg="#2e6fce", bg="#f8fbff")
    widgets["score_label"].config(text="Estimated average AQI score: —", bg="#f8fbff")
    widgets["advice_label"].config(text="Health recommendation will appear here.", bg="#f8fbff")
    widgets["status_label"].config(text="Application ready to predict air quality.", bg="#f8fbff")
    widgets["metrics_text"].set(get_metrics_summary(None))


def main():
    try:
        model, metrics = load_or_train_model()
    except FileNotFoundError:
        messagebox.showerror(
            "Dataset missing",
            f"Could not locate the dataset file at {DATA_PATH}. Please verify the dataset path and try again.",
        )
        sys.exit(1)

    root, entries, result_widgets = create_main_window(model, metrics)
    root.mainloop()


if __name__ == "__main__":
    main()
