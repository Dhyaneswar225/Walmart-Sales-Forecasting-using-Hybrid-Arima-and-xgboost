import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="Walmart Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
)

# Custom CSS Styling (Material / Neo-Morphism Style)
st.markdown("""
    <style>
    /* Sidebar Styling */
    section[data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #0f2027, #203a43, #2c5364);
        color: white;
    }
    .css-1lcbmhc, .css-1v0mbdj { color: white !important; }

    /* Smooth cards */
    .card {
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 3px 10px rgba(0,0,0,0.1);
        margin-bottom: 25px;
    }

    /* Section Titles */
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
        margin-top: 5px;
    }

    /* Dataframe Cleaner */
    .dataframe {
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        background-color: #2c5364;
        color: white;
        padding: 10px 18px;
        font-size: 16px;
    }

    /* Plots full width */
    .plot-container {
        padding: 8px;
        background: white;
        border-radius: 12px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 25px;
    }
    /* Sidebar Background */
    [data-testid="stSidebar"] {
            background-color: #0E1117; /* dark background */
            color: white;
    }
    /* Sidebar radio text */
    div[role="radiogroup"] > label > div {
            color: white !important;
            font-weight: 600 !important;
            font-size: 16px !important;
    }

    /* Selected radio option */
    div[role="radiogroup"] > label[data-baseweb="radio"] {
            color: white !important;
    }

    /* Hover effect */
    div[role="radiogroup"] > label:hover {
            background-color: rgba(255,255,255,0.1);
            border-radius: 6px;
    }
    /* Change the label "Go to:" color to white */
    .stRadio > label {
            color: white !important;
            font-weight: bold;
    }

    /* Change sidebar text color as well */
        section[data-testid="stSidebar"] .css-1y4p8pa {
            color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

RESULTS_DIR = "F:/Advanced Analytics/Project/WalmartSalesForecasting/results/"
DATA_PATH = "F:/Advanced Analytics/Project/WalmartSalesForecasting/data/merged_walmart.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    metrics = pd.read_csv(os.path.join(RESULTS_DIR, "phase6_metrics.csv"))
    forecasts = pd.read_csv(os.path.join(RESULTS_DIR, "phase6_forecasts_compare.csv"), parse_dates=["Date"])
    return df, metrics, forecasts

df, metrics_df, fc_df = load_data()

st.sidebar.title("📊 Walmart Forecasting")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["📈 Data Exploration", "🔮 ARIMA Model Results",
     "🤖 Hybrid Model Results", "📉 Error Comparison"]
)

if page.startswith("📈"):
    st.title("📈 Walmart Sales — Data Exploration")
    st.markdown("<div class='card'>Explore trends, patterns, and seasonality in Walmart sales.</div>", unsafe_allow_html=True)

    with st.container():
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

    store = st.selectbox("Select Store", sorted(df["Store"].unique()))
    dept = st.selectbox("Select Department", sorted(df["Dept"].unique()))

    ts = df[(df["Store"] == store) & (df["Dept"] == dept)].sort_values("Date")

    st.write("### 📊 Time Series Plot")
    fig = px.line(ts, x="Date", y="Weekly_Sales",
                  title=f"Weekly Sales (Store {store}, Dept {dept})",
                  template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.write("### 📉 Rolling Mean (12 Weeks)")
    ts["rolling"] = ts["Weekly_Sales"].rolling(12).mean()
    fig2 = px.line(ts, x="Date", y=["Weekly_Sales", "rolling"],
                   labels={"value": "Sales"},
                   title="Trend (Rolling Mean — 12 Weeks)",
                   template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

    st.write("### 🔁 Seasonal Comparison (Year-wise)")
    ts["Year"] = ts["Date"].dt.year
    ts["Week"] = ts["Date"].dt.isocalendar().week.astype(int)
    fig3 = px.line(ts, x="Week", y="Weekly_Sales", color="Year",
                   title="Seasonality Comparison Across Years",
                   template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

elif page.startswith("🔮"):
    st.title("🔮 ARIMA Forecasting Results")
    st.markdown("<div class='card'>Model-based linear forecasting using SARIMA.</div>", unsafe_allow_html=True)

    arima_insample = pd.read_csv(os.path.join(RESULTS_DIR, "arima_insample.csv"))
    arima_fc = pd.read_csv(os.path.join(RESULTS_DIR, "arima_forecast.csv"))

    st.subheader("📍 In-Sample Fit vs Actual")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=arima_insample["actual"], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(y=arima_insample["fitted"], mode='lines', name='ARIMA Fitted'))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧭 ARIMA Forecast (Next Weeks)")
    fig2 = px.line(arima_fc, y="forecast", title="ARIMA Forecast", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

    st.write("### ARIMA Forecast Output")
    st.dataframe(arima_fc)

elif page.startswith("🤖"):
    st.title("🤖 Hybrid ARIMA + XGBoost Forecast")
    st.markdown("<div class='card'>Combining linear + nonlinear models for improved forecasting accuracy.</div>", unsafe_allow_html=True)

    st.subheader("📌 Actual vs ARIMA vs Hybrid")
    fig = px.line(fc_df, x="Date", y=["actual", "arima", "hybrid"],
                  title="Hybrid Forecast Comparison", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🆚 ML-only vs Hybrid")
    fig2 = px.line(fc_df, x="Date", y=["ml_only", "hybrid"],
                   title="ML-only vs Hybrid Model", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

    img_path = os.path.join(RESULTS_DIR, "phase6_resid_feature_importance.png")
    if os.path.exists(img_path):
        st.subheader("🔥 XGBoost Feature Importance")
        st.image(img_path, use_column_width=True)

elif page.startswith("📉"):
    st.title("📉 Model Performance Comparison")
    st.markdown("<div class='card'>Error metrics for ARIMA, ML-only, and the Hybrid model.</div>", unsafe_allow_html=True)

    st.subheader("📊 Metrics Table")
    st.dataframe(metrics_df.style.highlight_min(color="lightgreen", axis=0))

    st.subheader("📊 Error Comparison Chart")
    fig = px.bar(metrics_df.reset_index(), x="index",
                 y=["RMSE", "MAE", "MAPE", "wMAPE"],
                 barmode="group",
                 title="Error Comparison",
                 template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.download_button("⬇️ Download Metrics CSV",
                       data=metrics_df.to_csv(),
                       file_name="metrics.csv")
