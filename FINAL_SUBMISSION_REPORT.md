# 🌫️ Pearls AQI Predictor — Final Submission Report
**Project Submission & Engineering Technical Document**  
*Organization*: 10Pearls | *Evaluation Timestamp*: September 06, 2026 - 17:48 UTC

---

## 1. Executive Summary & Serverless Architecture

The **Pearls AQI Predictor** is an autonomous, production-grade MLOps system delivering reliable, hourly **Air Quality Index (AQI)** forecasts up to 72 hours in advance. By unifying satellite and ground atmospheric observations from the **Open-Meteo API** with reproducible machine learning pipelines, the solution eliminates external API cost barriers while maintaining zero-crash fallback persistence.

### Architectural Pipeline Flow
```
[ Open-Meteo Air Quality & Weather API ] (No API Key, 90-Day Backfill + 72h Forecast)
                   │
                   ▼
[ 1_feature_pipeline.py ] ──► Dual-Mode Ingestion (Hopsworks Feature Store / data/*.parquet)
                   │
                   ▼
[ 2_training_pipeline.py ] ─► Multi-Model Benchmarking (Ridge vs. RF vs. TensorFlow DNN)
                   │
                   ▼
[ Deployment & Registry ] ──► Dual-Mode Registry (Hopsworks Registry / models/*.pkl & *.joblib)
                   │
                   ▼
[ 3_app.py / Docker ] ──────► Streamlit Interactive Visualizer & Plotly Risk Matrix
```

### Infrastructure Capabilities
- **Zero-Friction Ingestion**: Continuous ingestion of historical (90 days) and forecast horizons (72 hours) without rate limits or API keys.
- **Dual-Mode Persistence**: First-class integration with Hopsworks Feature Store and Model Registry via `HOPSWORKS_API_KEY`, with seamless local Parquet and Joblib/Pickle fallback when credentials are absent.
- **Continuous Integration / Continuous Deployment (CI/CD)**:
  - `feature_pipeline.yml`: Hourly ingestion trigger (`0 * * * *`).
  - `training_pipeline.yml`: Daily retraining trigger (`0 0 * * *`).

---

## 2. Telemetry & Feature Engineering Specification

The feature engineering pipeline transforms raw multi-source atmospheric data into a high-dimensional representation tailored for time-series forecasting.

### Dataset Overview
- **Total Ingested Historical Records**: `2,178` hourly samples
- **Temporal Horizon Covered**: `2026-06-08 00:00 UTC` through `2026-09-06 17:00 UTC`
- **Engineered Feature Dimensions**: `37` active predictive attributes

### Mathematical & Domain Transformations
1. **Particulate & Gas Telemetry**: High-frequency tracking of fine particulates ($PM_{2.5}$, $PM_{10}$) and trace gases ($NO_2$, $O_3$, $CO$, $SO_2$).
2. **Atmospheric Physics & Wind Vectors**: Trigonometric decomposition of wind speed ($V$) and direction ($	heta$) into orthogonal components:
   $$\text{wind}_u = -V \cdot \sin(\theta), \quad \text{wind}_v = -V \cdot \cos(\theta)$$
3. **Temporal Cyclical Encodings**: Non-linear trigonometric projection of cyclical diurnal and seasonal variations:
   $$\sin\left(\frac{2\pi \cdot h}{24}\right), \quad \cos\left(\frac{2\pi \cdot h}{24}\right), \quad \sin\left(\frac{2\pi \cdot d}{7}\right), \quad \cos\left(\frac{2\pi \cdot d}{7}\right)$$
4. **Temporal Lags & Inertia Windows**: Autoregressive lag features computed at intervals $t-1, t-2, t-3, t-6, t-12, t-24$ hours to capture atmospheric particulate inertia.
5. **Rolling Statistical Accumulators**: Moving-average windows (6h, 12h, 24h) and localized rolling standard deviations to capture volatility spikes.

---

## 3. Empirical Model Benchmarks & Selection

To ensure rigorous validation without temporal leakage, the dataset was partitioned chronologically into **75% Training**, **15% Validation**, and **10% Holdout Test** splits. Preprocessing pipelines (StandardScaler) were fitted exclusively on the training partition.

### Benchmark Evaluation Table
| Model Architecture | Test MAE | Test RMSE | Test R² Score | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Ridge** | 0.202 | 0.273 | 0.9921 | **🏆 Champion** |
| Random_Forest | 0.477 | 0.736 | 0.9424 | Evaluated |
| TensorFlow_DNN | 5.713 | 6.798 | -3.9154 | Evaluated |


### Champion Model Selection Rationale
The **Ridge** architecture was crowned champion:
- **Generalization Power**: Achieved the lowest test MAE of **0.202** and test RMSE of **0.273** with an $R^2$ of **0.9921**.
- **Inference Latency & Determinism**: Extremely lightweight footprint (<5 KB serialized), microsecond prediction latency, and absolute numeric stability under edge-case atmospheric anomalies.
- **Overfitting Resistance**: Strong L2 regularization penalty effectively dampens multicollinearity across dense particulate lag windows.

---

## 4. Explainability & Interpretability (XAI)

Model interpretability was audited using feature attribution rankings and verified via `models/shap_summary.png`.

![SHAP Summary](models/shap_summary.png)

### Key Explanatory Drivers
1. **`rolling_mean_6h_us_aqi`** (Importance Score: 4.2388)
2. **`lag_1h_us_aqi`** (Importance Score: 3.3888)
3. **`rolling_mean_24h_pm2_5`** (Importance Score: 1.7127)
4. **`lag_6h_us_aqi`** (Importance Score: 1.1470)
5. **`lag_2h_us_aqi`** (Importance Score: 0.8411)
6. **`lag_3h_us_aqi`** (Importance Score: 0.6676)
7. **`rolling_mean_24h_us_aqi`** (Importance Score: 0.4657)
8. **`lag_1h_pm2_5`** (Importance Score: 0.3085)


### Domain Analysis of Attribution
1. **Particulate Autoregressive Inertia**: Recent hourly lags (`lag_1h_us_aqi`, `rolling_mean_6h_us_aqi`) exert the strongest positive attribution on near-term predictions, reflecting real-world atmospheric persistence.
2. **Fine Particulate Correlation ($PM_{2.5}$)**: $PM_{2.5}$ serves as the primary chemical determinant in the US EPA AQI formula, directly driving hazardous classifications during smog conditions.
3. **Wind & Atmospheric Dispersion**: Wind vector components (`wind_u`, `wind_v`) and boundary pressure modulate whether pollutants accumulate or clear from urban micro-climates.

---

## 5. EPA Alert Matrix & Public Health Action Guidelines

| US AQI Range | Classification | Color Badge | Public Health Action Advisory |
| :---: | :---: | :---: | :--- |
| **0 – 50** | **Good** | Green (`#10B981`) | Air quality is satisfactory; enjoy normal outdoor activities. |
| **51 – 100** | **Moderate** | Yellow (`#F59E0B`) | Acceptable air quality; sensitive individuals should monitor exertion. |
| **101 – 150** | **Unhealthy for Sensitive** | Orange (`#F97316`) | Sensitive groups should reduce strenuous outdoor activities. |
| **151 – 200** | **Unhealthy** | Red (`#EF4444`) | Wear N95 masks outdoors, keep windows closed, and run HEPA purifiers. |
| **201 – 300** | **Very Unhealthy** | Purple (`#8B5CF6`) | Health alert: Avoid all outdoor exercise; stay in filtered environments. |
| **301 – 500** | **Hazardous** | Maroon (`#881337`) | Emergency warning: Serious cardiovascular/respiratory risk. Stay indoors. |

---

## 6. Reproducibility & Deployment Guide

### Local Pipeline Execution
```bash
# 1. Setup virtual environment and dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run feature extraction & model training
python 1_feature_pipeline.py --city Karachi
python 2_training_pipeline.py --city Karachi

# 3. Execute pre-submission audit suite
python audit_submission.py

# 4. Launch Streamlit UI
streamlit run 3_app.py
```

### Docker Containerized Deployment
```bash
# Build and run the production container
docker build -t pearls-aqi-dashboard:latest .
docker run -d -p 8501:8501 --name aqi-dashboard pearls-aqi-dashboard:latest

# Or launch via Docker Compose
docker compose up -d
```
Access dashboard at `http://localhost:8501`.
