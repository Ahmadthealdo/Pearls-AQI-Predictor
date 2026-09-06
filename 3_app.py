import os
import sys
import json
import logging
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import joblib
import pickle
from dotenv import load_dotenv

# Load environment variables at the very top
load_dotenv()

from api import fetch_live_aqicn, compare_prediction_with_live, get_aqicn_token

# Streamlit Page Configuration
st.set_page_config(
    page_title="Pearls AQI Predictor | 10Pearls",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS)
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .badge-good { background-color: #10B981; color: white; padding: 4px 12px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; }
    .badge-moderate { background-color: #F59E0B; color: white; padding: 4px 12px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; }
    .badge-sensitive { background-color: #F97316; color: white; padding: 4px 12px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; }
    .badge-unhealthy { background-color: #EF4444; color: white; padding: 4px 12px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; }
    .badge-very-unhealthy { background-color: #8B5CF6; color: white; padding: 4px 12px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; }
    .badge-hazardous { background-color: #881337; color: white; padding: 4px 12px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# Cities mapping
CITIES = {
    "Karachi": {"lat": 24.8607, "lon": 67.0011, "country": "Pakistan"},
    "Lahore": {"lat": 31.5497, "lon": 74.3436, "country": "Pakistan"},
    "Islamabad": {"lat": 33.6844, "lon": 73.0479, "country": "Pakistan"}
}


def get_aqi_category(aqi: float) -> tuple[str, str, str, str]:
    """Returns (category_name, badge_css_class, color_hex, health_recommendation)."""
    if aqi <= 50:
        return (
            "Good",
            "badge-good",
            "#10B981",
            "Air quality is satisfactory and poses little or no risk. Ideal for outdoor activities and natural ventilation."
        )
    elif aqi <= 100:
        return (
            "Moderate",
            "badge-moderate",
            "#F59E0B",
            "Air quality is acceptable. However, unusually sensitive individuals should consider limiting prolonged outdoor exertion."
        )
    elif aqi <= 150:
        return (
            "Unhealthy for Sensitive Groups",
            "badge-sensitive",
            "#F97316",
            "Children, older adults, and people with respiratory or heart disease should reduce prolonged outdoor exertion."
        )
    elif aqi <= 200:
        return (
            "Unhealthy",
            "badge-unhealthy",
            "#EF4444",
            "Everyone may begin to experience health effects. Wear N95 masks outdoors, keep windows closed, and run HEPA air filtration."
        )
    elif aqi <= 300:
        return (
            "Very Unhealthy",
            "badge-very-unhealthy",
            "#8B5CF6",
            "Health alert: risk of health effects increased for everyone. Avoid outdoor activities and remain in well-filtered indoor environments."
        )
    else:
        return (
            "Hazardous",
            "badge-hazardous",
            "#881337",
            "Emergency health warning: serious risk of respiratory and cardiovascular harm. Stay indoors, seal windows, and avoid physical strain."
        )


@st.cache_data(ttl=180)
def fetch_city_telemetry(city_name: str, lat: float, lon: float, past_days: int = 14, forecast_days: int = 3):
    """
    Dynamically fetches real-time Open-Meteo telemetry for the requested city and computes features.
    Falls back gracefully to local parquet files if offline.
    """
    try:
        import importlib
        fp = importlib.import_module("1_feature_pipeline")
        raw_df = fp.fetch_open_meteo_data(lat=lat, lon=lon, past_days=past_days, forecast_days=forecast_days)
        hist_df, forecast_df = fp.engineer_features(raw_df, city_name=city_name, lat=lat, lon=lon)
        return hist_df, forecast_df, "Live Open-Meteo API (Real-Time)"
    except Exception as e:
        # Fallback to local files if offline or network error
        hist_path = os.path.join("data", "latest_features.parquet")
        forecast_path = os.path.join("data", "aqi_forecast_features.parquet")
        if os.path.exists(hist_path) and os.path.exists(forecast_path):
            h_df = pd.read_parquet(hist_path)
            f_df = pd.read_parquet(forecast_path)
            h_df["time"] = pd.to_datetime(h_df["time"], utc=True)
            f_df["time"] = pd.to_datetime(f_df["time"], utc=True)
            return h_df, f_df, "Local Cache (Offline Fallback)"
        raise e


@st.cache_data(ttl=600)
def load_data(data_dir: str = "data"):
    """Loads historical features, forecast features, and metadata."""
    hist_path = os.path.join(data_dir, "latest_features.parquet") if os.path.exists(os.path.join(data_dir, "latest_features.parquet")) else os.path.join(data_dir, "aqi_features.parquet")
    forecast_path = os.path.join(data_dir, "aqi_forecast_features.parquet")
    meta_path = os.path.join(data_dir, "metadata.json")

    hist_df = pd.read_parquet(hist_path) if os.path.exists(hist_path) else None
    forecast_df = pd.read_parquet(forecast_path) if os.path.exists(forecast_path) else None
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)

    if hist_df is not None:
        hist_df["time"] = pd.to_datetime(hist_df["time"], utc=True)
    if forecast_df is not None:
        forecast_df["time"] = pd.to_datetime(forecast_df["time"], utc=True)

    return hist_df, forecast_df, metadata


@st.cache_resource
def load_champion_model(models_dir: str = "models"):
    """Loads champion model and metrics."""
    pkl_path = os.path.join(models_dir, "champion_model.pkl")
    joblib_path = os.path.join(models_dir, "aqi_model.joblib")
    model_path = pkl_path if os.path.exists(pkl_path) else joblib_path
    metrics_path = os.path.join(models_dir, "metrics.json")
    metadata_path = os.path.join(models_dir, "metadata.json")

    model = joblib.load(model_path) if os.path.exists(model_path) else None
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    return model, metrics, metadata


def generate_future_predictions(model, forecast_df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Predicts AQI for future forecast horizon using the trained champion model."""
    if model is None or forecast_df is None or len(forecast_df) == 0:
        return pd.DataFrame()

    available_features = [f for f in feature_names if f in forecast_df.columns]
    X_forecast = forecast_df[available_features].ffill().bfill()
    preds = model.predict(X_forecast)

    df_preds = forecast_df[["time", "city", "latitude", "longitude"]].copy()
    df_preds["predicted_aqi"] = np.round(np.clip(preds, 0, 500), 1)

    # Attach weather and pollutant signals for deep dive tooltips
    for col in ["pm2_5", "pm10", "temperature_2m", "relative_humidity_2m", "wind_speed_10m"]:
        if col in forecast_df.columns:
            df_preds[col] = forecast_df[col]

    return df_preds


# Main UI Layout
def main():
    # Sidebar
    st.sidebar.image("https://images.unsplash.com/photo-1534088568595-a066f410bcda?w=400&q=80", use_container_width=True)
    st.sidebar.markdown("### ⚙️ Pipeline & Location Settings")

    selected_city = st.sidebar.selectbox("Select Target City", list(CITIES.keys()), index=0)
    city_info = CITIES[selected_city]
    
    forecast_horizon = st.sidebar.slider("Forecast Horizon (Hours)", min_value=12, max_value=72, value=72, step=12)

    # AQICN Token Input field
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔑 Ground Station Verification")
    token_from_env = get_aqicn_token() or ""
    aqicn_tok = st.sidebar.text_input(
        "AQICN API Token",
        value=token_from_env,
        type="password",
        help="Token is read from AQICN_API_TOKEN in .env or enter here to enable live ground station comparison."
    )
    if aqicn_tok:
        st.sidebar.success("● AQICN Ground Truth: Active (Token Configured)")
    else:
        st.sidebar.info("● AQICN Ground Truth: Offline (No Token in .env)")

    # System Status & Hopsworks Indicator
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔌 MLOps Infrastructure")
    
    hw_key = os.getenv("HOPSWORKS_API_KEY", "").strip()
    if hw_key:
        st.sidebar.success("● Hopsworks Feature Store: Connected")
        st.sidebar.success("● Hopsworks Model Registry: Connected")
    else:
        st.sidebar.info("● Storage: Local Persistent Parquet")
        st.sidebar.info("● Model Registry: Local Joblib Artifacts")

    # Real-Time Fetch / Refresh controls
    st.sidebar.markdown("---")
    col_ref1, col_ref2 = st.sidebar.columns(2)
    with col_ref1:
        if st.button("⚡ Live Sync", use_container_width=True, help="Force fresh API telemetry sync"):
            st.cache_data.clear()
            st.rerun()
    with col_ref2:
        if st.button("🔄 Retrain", use_container_width=True, help="Retrain ML champion model"):
            with st.spinner("Retraining model..."):
                import subprocess
                subprocess.run([sys.executable, "2_training_pipeline.py", "--city", selected_city], check=True)
                st.cache_resource.clear()
                st.rerun()

    # Load dynamic real-time telemetry for THIS selected city
    with st.spinner(f"Connecting to live atmospheric telemetry for {selected_city}..."):
        hist_df, forecast_df, data_source = fetch_city_telemetry(selected_city, city_info["lat"], city_info["lon"])
    model, metrics, model_meta = load_champion_model()

    # Check if pipeline data exists
    if hist_df is None or model is None:
        st.warning("⚠️ Pipeline data or trained model weights were not found. Please run the feature and training pipelines to initialize the system.")
        if st.button("🚀 Initialize System Now"):
            with st.spinner("Executing feature extraction and model training..."):
                import subprocess
                subprocess.run([sys.executable, "1_feature_pipeline.py", "--city", selected_city], check=True)
                subprocess.run([sys.executable, "2_training_pipeline.py", "--city", selected_city], check=True)
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
        return

    # Header
    st.markdown('<div class="main-title">🌫️ Pearls AQI Predictor</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">Autonomous AI-powered Air Quality Index forecasting for <b>{selected_city}</b>, combining high-resolution satellite/sensor telemetry from Open-Meteo with machine learning. • Telemetry Source: <span style="color:#059669; font-weight:700;">{data_source}</span></div>', unsafe_allow_html=True)

    # Current Status Snapshot
    latest_hist = hist_df.iloc[-1]
    current_aqi = float(latest_hist.get("us_aqi", 50))
    cat_name, badge_class, cat_color, health_advice = get_aqi_category(current_aqi)

    col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1, 1, 1])

    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 6px solid {cat_color};">
            <span style="font-size: 0.85rem; color: #64748B; text-transform: uppercase; font-weight: 700;">Current AQI</span>
            <div style="display: flex; align-items: baseline; gap: 10px; margin: 4px 0;">
                <span style="font-size: 2.5rem; font-weight: 800; color: #1E293B;">{int(current_aqi)}</span>
                <span class="{badge_class}">{cat_name}</span>
            </div>
            <span style="font-size: 0.75rem; color: #94A3B8;">Last updated: {latest_hist['time'].strftime('%b %d, %H:%M UTC')}</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        pm25 = latest_hist.get("pm2_5", 0.0)
        st.metric("PM2.5 (Fine Particulate)", f"{pm25:.1f} µg/m³", delta=f"{pm25 - 15.0:.1f} vs WHO limit", delta_color="inverse")

    with col3:
        pm10 = latest_hist.get("pm10", 0.0)
        st.metric("PM10 (Coarse Particulate)", f"{pm10:.1f} µg/m³")

    with col4:
        temp = latest_hist.get("temperature_2m", 25.0)
        st.metric("Temperature", f"{temp:.1f} °C")

    with col5:
        humidity = latest_hist.get("relative_humidity_2m", 50.0)
        st.metric("Humidity", f"{humidity:.0f} %")

    # Health Advisory Alert
    st.info(f"💡 **Health Advisory ({cat_name})**: {health_advice}")

    # Real-Time Comparison Engine: Model Predictions vs. AQICN Ground Truth Sensor
    if aqicn_tok:
        live_res = fetch_live_aqicn(selected_city, lat=city_info["lat"], lon=city_info["lon"], token=aqicn_tok)
        if live_res.get("status") == "success" and live_res.get("data"):
            comp = compare_prediction_with_live(current_aqi, live_res["data"])
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%); border: 1px solid #86EFAC; border-radius: 10px; padding: 14px 20px; margin: 15px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <span style="font-weight: 800; color: #166534; font-size: 1.05rem;">📡 Real-Time Ground Truth Sensor Comparison:</span>
                        <span style="font-weight: 700; color: #1E293B;"> {comp['station_name']}</span>
                        <span style="color: #64748B; font-size: 0.85rem;"> (Reported: {comp['timestamp']})</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span style="font-weight: 800; font-size: 1.2rem; color: {comp['color']};">Ground AQI: {int(comp['live_aqi'])}</span>
                        <span style="background: white; border: 1px solid #BBF7D0; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.85rem; color: #1E293B;">
                            Residual Δ: {comp['delta']:+.1f} ({comp['alignment']})
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("ℹ️ *Add `AQICN_API_TOKEN` to `.env` to activate live physical sensor ground-truth comparison.*")

    # Generate Predictions
    feature_names = model_meta.get("features", [])
    predictions_df = generate_future_predictions(model, forecast_df, feature_names)
    if not predictions_df.empty:
        predictions_df = predictions_df.head(forecast_horizon)

    # Navigation Tabs
    tab_forecast, tab_history, tab_pollutants, tab_mlops = st.tabs([
        "📈 72-Hour AQI Forecast",
        "🕒 90-Day Historical Trend",
        "🔬 Pollutant Matrix",
        "⚙️ MLOps & Model Diagnostics"
    ])

    with tab_forecast:
        st.markdown("#### Hourly AQI Prediction Horizon")
        if not predictions_df.empty:
            fig = go.Figure()

            # AQI Severity Bands
            fig.add_hrect(y0=0, y1=50, fillcolor="#10B981", opacity=0.12, line_width=0, annotation_text="Good (0-50)", annotation_position="top left")
            fig.add_hrect(y0=50, y1=100, fillcolor="#F59E0B", opacity=0.12, line_width=0, annotation_text="Moderate (51-100)", annotation_position="top left")
            fig.add_hrect(y0=100, y1=150, fillcolor="#F97316", opacity=0.12, line_width=0, annotation_text="Unhealthy for Sensitive (101-150)", annotation_position="top left")
            fig.add_hrect(y0=150, y1=200, fillcolor="#EF4444", opacity=0.12, line_width=0, annotation_text="Unhealthy (151-200)", annotation_position="top left")
            fig.add_hrect(y0=200, y1=300, fillcolor="#8B5CF6", opacity=0.12, line_width=0, annotation_text="Very Unhealthy (201-300)", annotation_position="top left")
            fig.add_hrect(y0=300, y1=500, fillcolor="#881337", opacity=0.12, line_width=0, annotation_text="Hazardous (301+)", annotation_position="top left")

            # Forecast Line
            fig.add_trace(go.Scatter(
                x=predictions_df["time"],
                y=predictions_df["predicted_aqi"],
                mode="lines+markers",
                name="Predicted US AQI",
                line=dict(color="#1E3A8A", width=3),
                marker=dict(size=6, color="#3B82F6"),
                hovertemplate="<b>%{x|%a, %b %d, %H:%M UTC}</b><br>Predicted AQI: <b>%{y:.1f}</b><extra></extra>"
            ))

            fig.update_layout(
                title=f"Next {forecast_horizon} Hours Predicted Air Quality Index ({selected_city})",
                xaxis_title="Time (UTC)",
                yaxis_title="US AQI",
                yaxis=dict(range=[0, max(350, predictions_df["predicted_aqi"].max() + 30)]),
                template="plotly_white",
                height=450,
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Forecast Summary Highlights
            max_pred = predictions_df["predicted_aqi"].max()
            min_pred = predictions_df["predicted_aqi"].min()
            avg_pred = predictions_df["predicted_aqi"].mean()
            peak_time = predictions_df.loc[predictions_df["predicted_aqi"].idxmax(), "time"]

            f_col1, f_col2, f_col3 = st.columns(3)
            f_col1.metric("Predicted Peak AQI", f"{max_pred:.0f}", f"At {peak_time.strftime('%a %H:%M UTC')}")
            f_col2.metric("Predicted Average AQI", f"{avg_pred:.0f}")
            f_col3.metric("Predicted Cleanest AQI", f"{min_pred:.0f}")

    with tab_history:
        st.markdown("#### Historical AQI Observations (Past 90 Days)")
        days_history = st.slider("Historical Window (Days)", min_value=7, max_value=90, value=30, step=7)
        hist_subset = hist_df.iloc[-days_history * 24:].copy()

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=hist_subset["time"],
            y=hist_subset["us_aqi"],
            mode="lines",
            name="Observed US AQI",
            line=dict(color="#0284C7", width=1.5)
        ))
        
        if "rolling_mean_24h_us_aqi" in hist_subset.columns:
            fig_hist.add_trace(go.Scatter(
                x=hist_subset["time"],
                y=hist_subset["rolling_mean_24h_us_aqi"],
                mode="lines",
                name="24-Hour Moving Average",
                line=dict(color="#DC2626", width=2.5, dash="dash")
            ))

        fig_hist.update_layout(
            title=f"Historical AQI Trend (Past {days_history} Days)",
            xaxis_title="Date",
            yaxis_title="US AQI",
            template="plotly_white",
            height=420
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab_pollutants:
        st.markdown("#### Individual Atmospheric Pollutant Concentrations")
        pollutant_subset = hist_df.iloc[-7 * 24:].copy()

        fig_p = make_subplots(rows=2, cols=2, subplot_titles=["PM2.5 vs PM10", "Nitrogen Dioxide (NO2)", "Ozone (O3)", "Carbon Monoxide (CO)"])

        fig_p.add_trace(go.Scatter(x=pollutant_subset["time"], y=pollutant_subset["pm2_5"], name="PM2.5 (µg/m³)", line=dict(color="#EF4444")), row=1, col=1)
        fig_p.add_trace(go.Scatter(x=pollutant_subset["time"], y=pollutant_subset["pm10"], name="PM10 (µg/m³)", line=dict(color="#F97316")), row=1, col=1)

        if "nitrogen_dioxide" in pollutant_subset.columns:
            fig_p.add_trace(go.Scatter(x=pollutant_subset["time"], y=pollutant_subset["nitrogen_dioxide"], name="NO2 (µg/m³)", line=dict(color="#3B82F6")), row=1, col=2)

        if "ozone" in pollutant_subset.columns:
            fig_p.add_trace(go.Scatter(x=pollutant_subset["time"], y=pollutant_subset["ozone"], name="O3 (µg/m³)", line=dict(color="#10B981")), row=2, col=1)

        if "carbon_monoxide" in pollutant_subset.columns:
            fig_p.add_trace(go.Scatter(x=pollutant_subset["time"], y=pollutant_subset["carbon_monoxide"], name="CO (µg/m³)", line=dict(color="#6B7280")), row=2, col=2)

        fig_p.update_layout(height=600, template="plotly_white", showlegend=True)
        st.plotly_chart(fig_p, use_container_width=True)

    with tab_mlops:
        st.markdown("#### Production MLOps & Model Diagnostics")
        champ_name = metrics.get("champion_model", "Unknown")
        test_m = metrics.get("test_metrics", {})

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Champion Architecture", champ_name)
        m2.metric("Test MAE", f"{test_m.get('mae', 0.0)}")
        m3.metric("Test RMSE", f"{test_m.get('rmse', 0.0)}")
        m4.metric("Test R² Score", f"{test_m.get('r2', 0.0)}")

        st.markdown("---")
        st.markdown("##### Feature Importance Ranking")
        top_features = metrics.get("top_features", {})
        if top_features:
            df_fi = pd.DataFrame(list(top_features.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
            fig_fi = px.bar(df_fi, x="Importance", y="Feature", orientation="h", title="Top Predictive Features in Champion Model", color="Importance", color_continuous_scale="Blues")
            fig_fi.update_layout(template="plotly_white", height=450)
            st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown("##### Candidate Benchmark Comparison")
        benchmarks = metrics.get("benchmark_comparison", {})
        if benchmarks:
            bench_rows = []
            for m_name, m_res in benchmarks.items():
                bench_rows.append({
                    "Model": m_name,
                    "Val MAE": m_res["validation"]["mae"],
                    "Val RMSE": m_res["validation"]["rmse"],
                    "Val R²": m_res["validation"]["r2"],
                    "Test MAE": m_res["test"]["mae"],
                    "Test RMSE": m_res["test"]["rmse"],
                    "Test R²": m_res["test"]["r2"]
                })
            st.dataframe(pd.DataFrame(bench_rows), use_container_width=True)


if __name__ == "__main__":
    main()

