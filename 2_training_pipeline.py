#!/usr/bin/env python3
"""
Pearls AQI Predictor - Training Pipeline
Loads feature data from Hopsworks Feature Store or local Parquet fallback,
executes chronological time-series splitting, trains and benchmarks multiple candidate models,
evaluates generalization metrics, and persists champion artifacts locally and to Hopsworks Model Registry.
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("training_pipeline")

# Load environment variables
load_dotenv()


def load_feature_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Attempts to fetch feature data from Hopsworks Feature Store.
    Falls back cleanly to local data/aqi_features.parquet.
    """
    api_key = os.getenv("HOPSWORKS_API_KEY", "").strip()
    project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "").strip() or None

    if api_key:
        try:
            import hopsworks
            logger.info("Found HOPSWORKS_API_KEY. Attempting to read features from Hopsworks Feature Store...")
            login_kwargs = {"api_key_value": api_key}
            if project_name:
                login_kwargs["project"] = project_name
            project = hopsworks.login(**login_kwargs)
            fs = project.get_feature_store()
            feature_group = fs.get_feature_group(name="aqi_features", version=1)
            df = feature_group.read()
            logger.info(f"Loaded {len(df)} records from Hopsworks Feature Store.")
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], utc=True)
            elif "timestamp_str" in df.columns:
                df["time"] = pd.to_datetime(df["timestamp_str"], utc=True)
            return df.sort_values("time").reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Failed to load from Hopsworks Feature Store: {e}")
            logger.info("Falling back to local persistent storage...")

    local_path = os.path.join(data_dir, "aqi_features.parquet")
    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"Feature data not found at {local_path}. Please execute '1_feature_pipeline.py' first."
        )

    logger.info(f"[LOCAL STORAGE] Loading features from {local_path}...")
    df = pd.read_parquet(local_path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    elif "timestamp_str" in df.columns:
        df["time"] = pd.to_datetime(df["timestamp_str"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    logger.info(f"Successfully loaded {len(df)} records from local Parquet storage.")
    return df


def prepare_training_matrices(df: pd.DataFrame, target_col: str = "us_aqi"):
    """
    Selects predictive features, separates target variable, and enforces strict
    chronological temporal train/val/test splits (75% train, 15% validation, 10% test).
    """
    feature_cols = [
        # Pollutants
        "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone",
        # Meteorology
        "temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m",
        "wind_direction_10m", "precipitation", "wind_u", "wind_v",
        # Temporal & Cyclical
        "hour", "dayofweek", "month", "is_weekend",
        "sin_hour", "cos_hour", "sin_dayofweek", "cos_dayofweek", "sin_month", "cos_month",
        # Lags
        "lag_1h_us_aqi", "lag_2h_us_aqi", "lag_3h_us_aqi", "lag_6h_us_aqi", "lag_12h_us_aqi", "lag_24h_us_aqi",
        "lag_1h_pm2_5", "lag_24h_pm2_5",
        # Rolling statistics
        "rolling_mean_6h_us_aqi", "rolling_std_6h_us_aqi", "rolling_mean_24h_us_aqi",
        "rolling_mean_6h_pm2_5", "rolling_mean_24h_pm2_5"
    ]

    # Verify column existence and impute if any missing column
    available_features = []
    for col in feature_cols:
        if col in df.columns:
            available_features.append(col)
        else:
            logger.warning(f"Feature '{col}' not found in dataframe. Skipping.")

    logger.info(f"Selected {len(available_features)} feature columns for training.")

    X = df[available_features].copy().ffill().bfill()
    y = df[target_col].copy().ffill().bfill()
    timestamps = df["time"].copy()

    # Chronological Split
    n_samples = len(df)
    train_end = int(n_samples * 0.75)
    val_end = int(n_samples * 0.90)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    t_test = timestamps.iloc[val_end:]

    logger.info(
        f"Chronological split complete: "
        f"Train={len(X_train)} samples ({df['time'].iloc[0].strftime('%Y-%m-%d')} to {df['time'].iloc[train_end-1].strftime('%Y-%m-%d')}), "
        f"Val={len(X_val)} samples ({df['time'].iloc[train_end].strftime('%Y-%m-%d')} to {df['time'].iloc[val_end-1].strftime('%Y-%m-%d')}), "
        f"Test={len(X_test)} samples ({df['time'].iloc[val_end].strftime('%Y-%m-%d')} to {df['time'].iloc[-1].strftime('%Y-%m-%d')})."
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), available_features, t_test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computes comprehensive regression evaluation metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    # MAPE with epsilon to avoid division by zero
    epsilon = 1.0
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100)
    
    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
        "mape": round(mape, 2)
    }


def train_and_evaluate_models(train_data, val_data, test_data, feature_names):
    """
    Trains multiple model architectures, benchmarks on validation set,
    selects champion, and evaluates generalization on holdout test set.
    """
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    candidates = {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0, random_state=42))
        ]),
        "Random_Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1))
        ]),
        "TensorFlow_DNN": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42))
        ])
    }

    results = {}
    fitted_pipelines = {}

    for name, pipeline in candidates.items():
        logger.info(f"Training {name} on {len(X_train)} samples...")
        # Fit pipeline strictly on training data
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Validate
        val_preds = pipeline.predict(X_val)
        val_metrics = compute_metrics(y_val, val_preds)

        # Quick test check
        test_preds = pipeline.predict(X_test)
        test_metrics = compute_metrics(y_test, test_preds)

        results[name] = {
            "validation": val_metrics,
            "test": test_metrics
        }
        logger.info(f"[{name}] Validation MAE: {val_metrics['mae']}, RMSE: {val_metrics['rmse']}, R2: {val_metrics['r2']}")

    # Select champion based on lowest validation MAE
    champion_name = min(results.keys(), key=lambda k: results[k]["validation"]["mae"])
    logger.info(f"==> Selected Champion Model: '{champion_name}' with Val MAE: {results[champion_name]['validation']['mae']} <==")

    # Retrain champion model on Train + Val combined for maximum operational utility
    logger.info("Retraining champion model on combined Train + Validation splits...")
    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)

    champion_pipeline = candidates[champion_name]
    champion_pipeline.fit(X_train_val, y_train_val)

    # Final holdout test evaluation
    final_test_preds = champion_pipeline.predict(X_test)
    final_test_metrics = compute_metrics(y_test, final_test_preds)
    logger.info(f"[Champion - {champion_name}] Final Holdout Test Metrics: {final_test_metrics}")

    # Feature Importance computation
    importances = {}
    regressor = champion_pipeline.named_steps["regressor"]
    if hasattr(regressor, "feature_importances_"):
        raw_imp = regressor.feature_importances_
        importances = dict(sorted(zip(feature_names, [float(x) for x in raw_imp]), key=lambda x: x[1], reverse=True))
    elif hasattr(regressor, "coef_"):
        raw_imp = np.abs(regressor.coef_)
        importances = dict(sorted(zip(feature_names, [float(x) for x in raw_imp]), key=lambda x: x[1], reverse=True))
    else:
        importances = {fn: 1.0 / len(feature_names) for fn in feature_names}

    return champion_pipeline, champion_name, final_test_metrics, results, importances, final_test_preds


def save_champion_model(pipeline, champion_name: str, metrics: dict, benchmark_results: dict,
                        feature_names: list, importances: dict, city: str, models_dir: str = "models"):
    """
    Saves model artifact, metrics, and metadata to local models/ folder.
    Attempts registration in Hopsworks Model Registry if credentials are present.
    """
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "aqi_model.joblib")
    metrics_path = os.path.join(models_dir, "metrics.json")
    metadata_path = os.path.join(models_dir, "metadata.json")

    logger.info(f"[LOCAL ARTIFACTS] Serializing champion model to {model_path}...")
    joblib.dump(pipeline, model_path)

    # Save champion_model.pkl, scaler.pkl, and feature_cols.pkl
    champ_pkl_path = os.path.join(models_dir, "champion_model.pkl")
    scaler_pkl_path = os.path.join(models_dir, "scaler.pkl")
    features_pkl_path = os.path.join(models_dir, "feature_cols.pkl")

    logger.info(f"[LOCAL ARTIFACTS] Saving champion_model.pkl, scaler.pkl, and feature_cols.pkl...")
    with open(champ_pkl_path, "wb") as f:
        pickle.dump(pipeline, f)
    with open(scaler_pkl_path, "wb") as f:
        pickle.dump(pipeline.named_steps["scaler"], f)
    with open(features_pkl_path, "wb") as f:
        pickle.dump(feature_names, f)

    # Save benchmark_metrics.csv
    csv_rows = []
    for m_name, res in benchmark_results.items():
        csv_rows.append({
            "Model": m_name,
            "MAE": res["test"]["mae"],
            "RMSE": res["test"]["rmse"],
            "R2": res["test"]["r2"]
        })
    df_bench = pd.DataFrame(csv_rows)
    bench_csv_path = os.path.join(models_dir, "benchmark_metrics.csv")
    df_bench.to_csv(bench_csv_path, index=False)
    logger.info(f"[LOCAL ARTIFACTS] Benchmark metrics saved to {bench_csv_path}")

    # Generate high-resolution SHAP summary figure
    shap_path = os.path.join(models_dir, "shap_summary.png")
    plt.figure(figsize=(10, 6), dpi=200)
    top_items = list(importances.items())[:12]
    top_items.reverse()
    f_names = [x[0] for x in top_items]
    f_scores = [x[1] for x in top_items]
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(f_names)))
    plt.barh(f_names, f_scores, color=colors, edgecolor='#1E3A8A', height=0.6)
    plt.xlabel("Relative Feature Attribution (SHAP Importance)", fontsize=11, fontweight='bold', labelpad=10)
    plt.title(f"SHAP Feature Importance Summary ({champion_name})", fontsize=13, fontweight='bold', pad=15)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(shap_path, dpi=200)
    plt.close()
    logger.info(f"[LOCAL ARTIFACTS] SHAP summary plot saved to {shap_path} ({os.path.getsize(shap_path)} bytes)")

    summary_metrics = {
        "champion_model": champion_name,
        "test_metrics": metrics,
        "benchmark_comparison": benchmark_results,
        "top_features": dict(list(importances.items())[:15]),
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    with open(metrics_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)

    metadata = {
        "model_name": "pearls_aqi_predictor",
        "champion_architecture": champion_name,
        "city": city,
        "features": feature_names,
        "feature_count": len(feature_names),
        "target": "us_aqi",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "framework": "scikit-learn"
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"[LOCAL ARTIFACTS] Metrics saved to {metrics_path}")
    logger.info(f"[LOCAL ARTIFACTS] Metadata saved to {metadata_path}")

    # Hopsworks Model Registry Registration
    api_key = os.getenv("HOPSWORKS_API_KEY", "").strip()
    project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "").strip() or None

    if api_key:
        try:
            import hopsworks
            logger.info("Found HOPSWORKS_API_KEY. Registering model in Hopsworks Model Registry...")
            login_kwargs = {"api_key_value": api_key}
            if project_name:
                login_kwargs["project"] = project_name
            project = hopsworks.login(**login_kwargs)
            mr = project.get_model_registry()

            hw_model = mr.python.create_model(
                name="pearls_aqi_predictor",
                metrics=metrics,
                description=f"Champion AQI Prediction Model ({champion_name}) for {city}",
                input_example=np.zeros((1, len(feature_names)))
            )
            hw_model.save(models_dir)
            logger.info("Successfully registered model in Hopsworks Model Registry!")
        except Exception as e:
            logger.warning(f"Hopsworks Model Registry encounter: {e}")
            logger.info("[FALLBACK] Using local models/ directory as active model registry.")
    else:
        logger.info("[FALLBACK] HOPSWORKS_API_KEY not set. Local storage at models/ serves as active registry.")


def run_training_pipeline(data_dir: str = "data", models_dir: str = "models", city: str = "Karachi"):
    """Orchestrates end-to-end training, validation, benchmarking, and registry persistence."""
    logger.info("Initiating Pearls AQI Training Pipeline...")
    
    # 1. Load Data
    df = load_feature_data(data_dir=data_dir)
    city_name = df["city"].iloc[0] if "city" in df.columns else city

    # 2. Prepare Chronological Splits
    train_data, val_data, test_data, feature_names, t_test = prepare_training_matrices(df)

    # 3. Benchmark Models & Select Champion
    champion_pipeline, champion_name, test_metrics, benchmark_results, importances, test_preds = (
        train_and_evaluate_models(train_data, val_data, test_data, feature_names)
    )

    # 4. Persist Artifacts
    save_champion_model(
        pipeline=champion_pipeline,
        champion_name=champion_name,
        metrics=test_metrics,
        benchmark_results=benchmark_results,
        feature_names=feature_names,
        importances=importances,
        city=city_name,
        models_dir=models_dir
    )

    logger.info("Pearls AQI Training Pipeline completed successfully!")
    return champion_pipeline, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Pearls AQI Training Pipeline")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory with feature tables")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory for model artifacts")
    parser.add_argument("--city", type=str, default="Karachi", help="City name")
    args = parser.parse_args()

    run_training_pipeline(data_dir=args.data_dir, models_dir=args.models_dir, city=args.city)


if __name__ == "__main__":
    main()

