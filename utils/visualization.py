import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from utils.data_utils import FEATURE_COLUMNS, TARGET_COLUMN


def build_aqi_trend_chart(df: pd.DataFrame):
    fig = px.line(
        df.reset_index(),
        x=df.index,
        y="AQI Value",
        title="AQI Trend Analysis",
        labels={"index": "Sample Index", "AQI Value": "AQI Value"},
        template="plotly_white",
    )
    fig.update_traces(line_color="#0d3b66", line_width=3)
    fig.update_layout(margin=dict(t=50, r=20, l=20, b=20))
    return fig


def build_pollutant_bar_chart(df: pd.DataFrame):
    averages = df[FEATURE_COLUMNS].mean().reset_index()
    averages.columns = ["Pollutant", "Average AQI"]
    fig = px.bar(
        averages,
        x="Pollutant",
        y="Average AQI",
        color="Pollutant",
        title="Average Pollutant AQI Comparison",
        template="plotly_white",
    )
    fig.update_layout(showlegend=False, margin=dict(t=50, r=20, l=20, b=20))
    return fig


def build_category_count_chart(df: pd.DataFrame):
    counts = df[TARGET_COLUMN].value_counts().reset_index()
    counts.columns = ["AQI Category", "Count"]
    fig = px.bar(
        counts,
        x="AQI Category",
        y="Count",
        color="AQI Category",
        title="AQI Category Frequency",
        template="plotly_white",
    )
    fig.update_layout(showlegend=False, margin=dict(t=50, r=20, l=20, b=20))
    return fig


def build_feature_importance_chart(feature_names, importances):
    fig = px.bar(
        x=importances,
        y=feature_names,
        orientation="h",
        title="Model Feature Importance",
        labels={"x": "Importance", "y": "Feature"},
        template="plotly_white",
    )
    fig.update_layout(margin=dict(t=50, r=20, l=20, b=20))
    fig.update_traces(marker_color="#f6b042")
    return fig


def build_pollutant_comparison_chart(inputs: dict):
    pollutants = list(inputs.keys())
    values = [inputs[name] for name in pollutants]
    fig = go.Figure(
        data=[go.Scatterpolar(r=values, theta=pollutants, fill="toself", name="Current Input")]
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(values) * 1.2])),
        showlegend=False,
        title="Pollutant Input Comparison",
        template="plotly_white",
        margin=dict(t=50, r=20, l=20, b=20),
    )
    return fig
