from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data_utils import (
    FEATURE_COLUMNS,
    AQI_STYLES,
    get_health_recommendation,
    get_sample_dataset,
    load_dataset,
    prepare_training_data,
)
from utils.model_utils import (
    load_model,
    save_model,
    train_and_evaluate,
    train_classifier,
    evaluate_model,
)
from utils.visualization import (
    build_aqi_trend_chart,
    build_category_count_chart,
    build_feature_importance_chart,
    build_pollutant_bar_chart,
    build_pollutant_comparison_chart,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "dataset" / "global_air_pollution_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "air_quality_model.pkl"

st.set_page_config(
    page_title="AI-Powered Air Quality Prediction Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles():
    st.markdown(
        """
        <style>
        .hero {
            background: linear-gradient(135deg, #0f4c81 0%, #3b82f6 100%);
            color: white;
            padding: 36px;
            border-radius: 24px;
            box-shadow: 0 35px 90px rgba(15, 46, 71, 0.18);
            margin-bottom: 24px;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.8rem;
            letter-spacing: -0.03em;
        }
        .hero p {
            margin-top: 12px;
            font-size: 1.05rem;
            color: rgba(255,255,255,0.85);
        }
        .panel-card {
            border-radius: 20px;
            background: rgba(255,255,255,0.92);
            backdrop-filter: blur(12px);
            box-shadow: 0 18px 45px rgba(15, 46, 71, 0.08);
            border: 1px solid rgba(255,255,255,0.48);
            padding: 24px;
            margin-bottom: 20px;
        }
        .metric-box {
            border-radius: 18px;
            padding: 20px;
            background: #ffffff;
            box-shadow: 0 12px 30px rgba(15, 46, 71, 0.08);
            text-align: center;
        }
        .metric-box h3 {
            margin: 0 0 10px 0;
            font-size: 1rem;
            color: #334e68;
        }
        .metric-box p {
            margin: 0;
            font-size: 1.9rem;
            font-weight: 700;
            color: #102a43;
        }
        .gradient-button {
            background: linear-gradient(135deg, #2dd4bf 0%, #22c55e 100%);
            border: none;
            color: white;
            padding: 0.8rem 1.4rem;
            border-radius: 999px;
            font-weight: 700;
            letter-spacing: 0.02em;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .gradient-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 18px 40px rgba(34, 197, 94, 0.26);
        }
        .category-chip {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 14px;
            border-radius: 999px;
            font-weight: 700;
            color: white;
            margin-bottom: 16px;
        }
        .category-chip.good { background: #22c55e; }
        .category-chip.moderate { background: #eab308; }
        .category-chip.poor { background: #f97316; }
        .category-chip.severe { background: #ef4444; }
        .small-note {
            color: #64748b;
            font-size: 0.95rem;
        }
        .sidebar-content {
            font-size: 0.95rem;
            color: #334e68;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=600)
def cached_load_dataset(path: str) -> pd.DataFrame:
    return load_dataset(Path(path))


@st.cache_resource
def cached_load_or_train_model(data_path: str, model_path: str):
    model_path_obj = Path(model_path)
    data_path_obj = Path(data_path)

    if model_path_obj.exists():
        try:
            model = load_model(model_path_obj)
            df = cached_load_dataset(str(data_path_obj)) if data_path_obj.exists() else get_sample_dataset()
            return model, None, df
        except Exception:
            pass

    try:
        model, metrics, df = train_and_evaluate(data_path_obj, model_path_obj)
        return model, metrics, df
    except FileNotFoundError:
        df = get_sample_dataset()
        X, y = prepare_training_data(df)
        model = train_classifier(X, y)
        metrics = evaluate_model(model, X, y)
        save_model(model, model_path_obj)
        return model, metrics, df
    except Exception as error:
        st.error(f"An unexpected error occurred while preparing the model: {error}")
        raise


def build_status_banner(category: str):
    if category == "Good":
        return "good", "Excellent air quality. Enjoy outdoor activities.", "success"
    if category == "Moderate":
        return "moderate", "Moderate air quality. Sensitive individuals should be cautious.", "info"
    if category == "Poor":
        return "poor", "Poor air quality. Consider limiting outdoor exposure.", "warning"
    return "severe", "Severe air quality. Stay indoors and limit exposure.", "error"


def format_report(inputs: dict, predicted_category: str, predicted_score: float) -> pd.DataFrame:
    report_data = {
        "PM2.5 AQI Value": [inputs["PM2.5 AQI Value"]],
        "PM10 AQI Value": [inputs["PM10 AQI Value"]],
        "NO2 AQI Value": [inputs["NO2 AQI Value"]],
        "SO2 AQI Value": [inputs["SO2 AQI Value"]],
        "CO AQI Value": [inputs["CO AQI Value"]],
        "Ozone AQI Value": [inputs["Ozone AQI Value"]],
        "Predicted AQI Score": [predicted_score],
        "Predicted AQI Category": [predicted_category],
        "Recommendation": [get_health_recommendation(predicted_category)],
    }
    return pd.DataFrame(report_data)


def get_category_color(category: str) -> str:
    return {
        "Good": "#22c55e",
        "Moderate": "#eab308",
        "Poor": "#f97316",
        "Very Poor": "#f43f5e",
        "Severe": "#ef4444",
    }.get(category, "#64748b")


def build_gauge_chart(category: str, score: float):
    color = get_category_color(category)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            title={"text": "AQI Meter", "font": {"size": 20}},
            delta={"reference": 100, "increasing": {"color": color}},
            gauge={
                "axis": {"range": [0, 500], "tickwidth": 1, "tickcolor": "#334e68"},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 50], "color": "#22c55e"},
                    {"range": [51, 100], "color": "#facc15"},
                    {"range": [101, 200], "color": "#fb923c"},
                    {"range": [201, 300], "color": "#f43f5e"},
                    {"range": [301, 500], "color": "#9d174d"},
                ],
            },
        )
    )
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def build_summary_cards(df: pd.DataFrame):
    total_samples = len(df)
    average_aqi = df["AQI Value"].mean() if "AQI Value" in df.columns else np.mean(df[FEATURE_COLUMNS].mean())
    category_counts = df["AQI Category"].value_counts().to_dict()
    top_sample = df.sort_values(by="AQI Value", ascending=False).head(1)
    most_polluted = (
        f"Sample #{int(top_sample.index[0])} - {top_sample['AQI Category'].values[0]}"
        if not top_sample.empty
        else "No data available"
    )
    return total_samples, average_aqi, most_polluted, category_counts


def render_sidebar() -> str:
    st.sidebar.markdown("# 🌿 AQI Dashboard")
    page = st.sidebar.radio(
        "Sections",
        ["Home", "Prediction", "Analytics", "Health", "About"],
        index=0,
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div class='sidebar-content'>Use this dashboard to explore pollution trends, run AI forecasts, and download prediction reports.</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    return page


def render_homepage(dataset: pd.DataFrame, metrics: dict | None):
    st.markdown(
        """
        <div class='hero'>
            <h1>AI-Powered Air Quality Prediction Dashboard</h1>
            <p class='small-note'>Real-time pollution analytics and machine learning powered AQI forecasting.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_samples, average_aqi, most_polluted, category_counts = build_summary_cards(dataset)
    stats = [
        ("Total Samples", total_samples, "📦"),
        ("Average AQI", f"{average_aqi:.1f}", "📈"),
        ("Most Polluted", most_polluted, "🌫️"),
        ("AQI Categories", ", ".join(category_counts.keys()), "🟢"),
    ]

    stat_cols = st.columns(4)
    for idx, (title, value, icon) in enumerate(stats):
        stat_cols[idx].markdown(
            f"<div class='metric-box'><h3>{icon} {title}</h3><p>{value}</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.markdown("### Overview")
    st.write(
        "This premium dashboard presents AQI forecasting, pollutant comparison, and health recommendations in a clean, modern analytics layout."
    )
    if metrics is not None:
        st.write(f"**Model Accuracy:** {metrics['accuracy'] * 100:.2f}%  |  **Cross Validation:** {metrics['cross_validation_score'] * 100:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.plotly_chart(build_aqi_trend_chart(dataset), use_container_width=True)
    with chart_cols[1]:
        st.plotly_chart(build_pollutant_bar_chart(dataset), use_container_width=True)

    st.markdown("---")
    st.subheader("AQI Categories")
    category_cols = st.columns(4)
    for idx, (label, color) in enumerate([
        ("Good", "#22c55e"),
        ("Moderate", "#eab308"),
        ("Poor", "#f97316"),
        ("Severe", "#ef4444"),
    ]):
        category_cols[idx].markdown(
            f"<div class='panel-card' style='background:{color}; color:white;'><h3>{label}</h3></div>",
            unsafe_allow_html=True,
        )


def render_prediction_page(model):
    st.markdown("<div class='panel-card'><h2>Real-Time AQI Prediction</h2><p class='small-note'>Enter pollutant AQI values to forecast air quality instantly.</p></div>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        inputs_left, inputs_right = st.columns(2)
        with inputs_left:
            pm25 = st.number_input("PM2.5", min_value=0.0, max_value=500.0, value=25.0, step=0.1)
            so2 = st.number_input("SO2", min_value=0.0, max_value=300.0, value=18.0, step=0.1)
            co = st.number_input("CO", min_value=0.0, max_value=200.0, value=12.0, step=0.1)
        with inputs_right:
            pm10 = st.number_input("PM10", min_value=0.0, max_value=500.0, value=45.0, step=0.1)
            no2 = st.number_input("NO2", min_value=0.0, max_value=300.0, value=30.0, step=0.1)
            ozone = st.number_input("Ozone", min_value=0.0, max_value=500.0, value=28.0, step=0.1)

        submit = st.form_submit_button("Predict Air Quality")

    if submit:
        pollutant_inputs = {
            "PM2.5 AQI Value": pm25,
            "PM10 AQI Value": pm10,
            "NO2 AQI Value": no2,
            "SO2 AQI Value": so2,
            "CO AQI Value": co,
            "Ozone AQI Value": ozone,
        }
        user_features = pd.DataFrame([pollutant_inputs], columns=FEATURE_COLUMNS)
        predicted_category = model.predict_category(user_features)[0]
        predicted_score = float(np.mean(user_features.iloc[0].values))
        report_df = format_report(pollutant_inputs, predicted_category, predicted_score)

        category_class, recommendation_text, alert_type = build_status_banner(predicted_category)
        if alert_type == "warning":
            st.warning(f"**{predicted_category} air quality detected.** {recommendation_text}")
        elif alert_type == "error":
            st.error(f"**{predicted_category} air quality detected.** {recommendation_text}")
        else:
            st.success(f"**{predicted_category} air quality detected.** {recommendation_text}")

        result_cols = st.columns([2, 3])
        with result_cols[0]:
            st.markdown(
                f"<div class='panel-card'><div class='category-chip {category_class}'>{predicted_category}</div><h3>Your AQI Forecast</h3><p style='font-size:2.3rem; margin:0;'>{predicted_score:.1f}</p><p class='small-note'>Higher values indicate poorer air quality.</p></div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(build_gauge_chart(predicted_category, predicted_score), use_container_width=True)

        with result_cols[1]:
            st.plotly_chart(build_pollutant_comparison_chart(pollutant_inputs), use_container_width=True)
            csv_report = report_df.to_csv(index=False)
            st.download_button(
                "Download Prediction Report",
                data=csv_report,
                file_name="aqi_prediction_report.csv",
                mime="text/csv",
                key="download-report",
            )


def render_analytics_page(dataset, metrics, model):
    st.markdown("<div class='panel-card'><h2>Analytics & Trends</h2><p class='small-note'>Visualize pollutant impact, AQI trends, and feature importance.</p></div>", unsafe_allow_html=True)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.plotly_chart(build_aqi_trend_chart(dataset), use_container_width=True)
    with chart_cols[1]:
        st.plotly_chart(build_pollutant_bar_chart(dataset), use_container_width=True)

    st.plotly_chart(build_category_count_chart(dataset), use_container_width=True)
    if metrics is not None:
        importance = model.pipeline.named_steps["classifier"].feature_importances_
        st.plotly_chart(build_feature_importance_chart(FEATURE_COLUMNS, importance), use_container_width=True)

    if metrics is not None:
        st.markdown("<div class='panel-card'><h3>Model Evaluation</h3><p class='small-note'>Current AI model performance and training summary.</p></div>", unsafe_allow_html=True)
        st.write(
            f"**Accuracy:** {metrics['accuracy'] * 100:.2f}%  |  **Cross-validation:** {metrics['cross_validation_score'] * 100:.2f}%"
        )


def render_health_page():
    st.markdown("<div class='panel-card'><h2>Health Recommendations</h2><p class='small-note'>Safety guidance based on AQI category and pollutant exposure.</p></div>", unsafe_allow_html=True)
    st.markdown(
        """
        - ✅ **Good:** Outdoor activity is safe. Maintain normal routines.
        - ⚠️ **Moderate:** Sensitive groups should reduce prolonged outdoor exertion.
        - 🔶 **Poor:** Avoid extended outdoor activity and use indoor air filtration.
        - 🔴 **Severe:** Stay indoors, close windows, and use air purifiers.
        """
    )
    st.markdown("---")
    st.markdown(
        "**Future health improvements:** GPS-based alerts, live AQI API feeds, and wearable pollution monitoring for smart cities."
    )


def render_about_page():
    st.markdown("<div class='panel-card'><h2>About This Project</h2><p class='small-note'>A polished AI-driven air quality dashboard built for GitHub portfolios and placement interviews.</p></div>", unsafe_allow_html=True)

    st.markdown(
        """
        The **AI-Powered Air Quality Prediction Dashboard** is a machine learning-based analytics platform designed to predict and analyze air quality using environmental pollutant data.

        This project provides real-time AQI insights, pollution trend analysis, and health recommendations through an interactive and modern Streamlit dashboard.

        Using machine learning algorithms and data visualization techniques, the system predicts air quality categories based on pollutant levels such as **PM2.5, PM10, NO2, SO2, CO, and Ozone**.
        """
    )

    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.subheader("🚀 Key Highlights")
    st.markdown(
        """
        - Real-time AQI prediction
        - Interactive analytics dashboard
        - Pollution trend visualization
        - Health and safety recommendations
        - Machine learning-based forecasting
        - Modern responsive UI using Streamlit
        """
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.subheader("🛠️ Technologies Used")
    st.markdown(
        """
        - Python
        - Streamlit
        - Scikit-learn
        - Pandas
        - NumPy
        - Plotly
        - Machine Learning
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.subheader("🎯 Project Goal")
    st.markdown(
        """
        The objective of this project is to create an intelligent and accessible platform that helps users understand pollution levels, identify environmental risks, and promote awareness regarding air quality and public health.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.subheader("👩‍💻 Developed by Vaidehi Sharma")
    st.markdown("A polished Streamlit dashboard experience tailored for both data-driven insights and portfolio presentation.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Technologies")
    st.write("Python, Streamlit, Plotly, Pandas, NumPy, Scikit-Learn, Joblib")


def main():
    inject_styles()
    model, metrics, dataset = cached_load_or_train_model(str(DATA_PATH), str(MODEL_PATH))
    page = render_sidebar()

    if page == "Home":
        render_homepage(dataset, metrics)
    elif page == "Prediction":
        render_prediction_page(model)
    elif page == "Analytics":
        render_analytics_page(dataset, metrics, model)
    elif page == "Health":
        render_health_page()
    elif page == "About":
        render_about_page()


if __name__ == "__main__":
    main()
