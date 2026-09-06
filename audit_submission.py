#!/usr/bin/env python3
"""
Pearls AQI Predictor - Comprehensive Pre-Submission Audit
Validates data artifacts, model artifacts, explainability plots, CI/CD workflows,
and headless Streamlit inference readiness against grading rubrics.
"""

import os
import sys
import pickle
import yaml
import numpy as np
import pandas as pd
import joblib

def print_banner(text: str):
    print("\n" + "=" * 75)
    print(f" {text.upper()}")
    print("=" * 75)


def audit_data_artifacts():
    print_banner("1. Data Artifacts Audit")
    file_path = os.path.join("data", "latest_features.parquet")
    
    assert os.path.exists(file_path), f"Missing required file: {file_path}"
    print(f"  [PASS] Found '{file_path}'.")

    df = pd.read_parquet(file_path)
    row_count = len(df)
    assert row_count > 1000, f"Expected >1000 rows, found {row_count}"
    print(f"  [PASS] Record count check passed ({row_count} rows > 1000).")

    # Datetime validation
    assert "time" in df.columns or "timestamp_str" in df.columns, "Missing timestamp column in dataset."
    time_col = "time" if "time" in df.columns else "timestamp_str"
    ts_series = pd.to_datetime(df[time_col], errors="coerce")
    assert ts_series.notnull().all(), "Found invalid or unparseable timestamps."
    print(f"  [PASS] Valid datetime timestamps verified ({ts_series.iloc[0]} to {ts_series.iloc[-1]}).")

    # Feature columns NaN check (excluding optional future target if applicable)
    feature_cols = [c for c in df.columns if c not in ["time", "timestamp_str", "timestamp_ms", "city"]]
    nan_counts = df[feature_cols].isna().sum()
    unexpected_nans = nan_counts[nan_counts > 0]
    assert len(unexpected_nans) == 0, f"Unexpected NaNs in features: {unexpected_nans.to_dict()}"
    print(f"  [PASS] Zero NaNs detected across all {len(feature_cols)} feature columns.")
    return True


def audit_model_artifacts():
    print_banner("2. Model Artifacts Audit")
    artifacts = {
        "champion_model.pkl": "Champion model object",
        "scaler.pkl": "Fitted standard scaler",
        "feature_cols.pkl": "Predictive feature column list",
        "benchmark_metrics.csv": "Multi-model benchmark evaluation table"
    }

    for fname, desc in artifacts.items():
        fpath = os.path.join("models", fname)
        assert os.path.exists(fpath), f"Missing model artifact: {fpath} ({desc})"
        print(f"  [PASS] Found '{fpath}'.")

    # Load champion_model.pkl
    champ_path = os.path.join("models", "champion_model.pkl")
    with open(champ_path, "rb") as f:
        model = pickle.load(f)
    assert hasattr(model, "predict"), "champion_model.pkl does not have a predict method."
    print("  [PASS] Successfully loaded 'champion_model.pkl' into memory.")

    # Load scaler.pkl
    scaler_path = os.path.join("models", "scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    assert hasattr(scaler, "transform"), "scaler.pkl is not a valid transformer."
    print("  [PASS] Successfully loaded 'scaler.pkl' into memory.")

    # Load feature_cols.pkl
    cols_path = os.path.join("models", "feature_cols.pkl")
    with open(cols_path, "rb") as f:
        cols = pickle.load(f)
    assert isinstance(cols, list) and len(cols) > 0, "feature_cols.pkl is not a non-empty list."
    print(f"  [PASS] Successfully loaded 'feature_cols.pkl' ({len(cols)} features registered).")

    # Load benchmark_metrics.csv
    bench_path = os.path.join("models", "benchmark_metrics.csv")
    df_bench = pd.read_csv(bench_path)
    assert not df_bench.empty, "benchmark_metrics.csv is empty."
    required_cols = {"Model", "MAE", "RMSE", "R2"}
    assert required_cols.issubset(set(df_bench.columns)), f"benchmark_metrics.csv missing columns. Required: {required_cols}"
    print(f"  [PASS] Validated 'benchmark_metrics.csv' containing {len(df_bench)} benchmarked architectures.")
    return True


def audit_interpretability():
    print_banner("3. Explainability & Interpretability Audit")
    shap_path = os.path.join("models", "shap_summary.png")
    assert os.path.exists(shap_path), f"Missing interpretability plot: {shap_path}"

    size_bytes = os.path.getsize(shap_path)
    assert size_bytes > 10240, f"shap_summary.png is smaller than 10 KB ({size_bytes} bytes)."
    print(f"  [PASS] 'shap_summary.png' exists and is non-empty ({size_bytes / 1024:.1f} KB > 10 KB).")
    return True


def audit_cicd_infrastructure():
    print_banner("4. CI/CD Workflows Audit")
    workflows = {
        "feature_pipeline.yml": ("0 * * * *", "hourly feature extraction"),
        "training_pipeline.yml": ("0 0 * * *", "daily model retraining")
    }

    for w_name, (expected_cron, desc) in workflows.items():
        w_path = os.path.join(".github", "workflows", w_name)
        assert os.path.exists(w_path), f"Missing CI/CD workflow file: {w_path}"

        with open(w_path, "r") as f:
            content = f.read()
            data = yaml.safe_load(content)

        assert data is not None, f"Workflow '{w_name}' contains empty or invalid YAML."

        on_block = data.get("on") or data.get(True) or {}
        schedules = on_block.get("schedule", [])
        crons = [s.get("cron") for s in schedules if isinstance(s, dict) and "cron" in s]
        assert expected_cron in crons, f"Expected cron '{expected_cron}' ({desc}) not found in '{w_name}'. Found: {crons}"
        print(f"  [PASS] Workflow '{w_name}' has valid YAML and required cron schedule '{expected_cron}'.")

    return True


def audit_dashboard_headless_inference():
    print_banner("5. Dashboard Headless Inference Audit")
    # Dynamically import helper functions from 3_app.py
    import importlib.util
    spec = importlib.util.spec_from_file_location("app_module", "3_app.py")
    app = importlib.util.module_from_spec(spec)
    
    # Mock streamlit cache functions if needed during headless import
    sys.modules["app_module"] = app
    spec.loader.exec_module(app)
    
    # Load model and feature cols
    with open(os.path.join("models", "champion_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join("models", "feature_cols.pkl"), "rb") as f:
        feature_cols = pickle.load(f)

    # Generate 72 simulated future hours
    now = pd.Timestamp.now(tz="UTC")
    future_times = [now + pd.Timedelta(hours=i) for i in range(1, 73)]
    
    mock_data = {
        "time": future_times,
        "city": ["Karachi"] * 72,
        "latitude": [24.8607] * 72,
        "longitude": [67.0011] * 72
    }
    for col in feature_cols:
        mock_data[col] = np.random.uniform(10.0, 50.0, size=72)

    df_forecast_sim = pd.DataFrame(mock_data)

    # Run app inference function
    preds_df = app.generate_future_predictions(model, df_forecast_sim, feature_cols)
    assert not preds_df.empty, "Inference returned an empty dataframe."
    assert len(preds_df) == 72, f"Expected 72 hourly predictions, got {len(preds_df)}."

    preds = preds_df["predicted_aqi"].values
    assert not np.isnan(preds).any(), "Predictions contain NaN values."
    assert not np.isinf(preds).any(), "Predictions contain infinite values."
    assert (preds >= 0).all() and (preds <= 500).all(), f"Predictions out of physical bounds (0-500). Min: {preds.min()}, Max: {preds.max()}"
    print(f"  [PASS] Headless 72-hour forecast validated successfully (Range: [{preds.min():.1f}, {preds.max():.1f}] US AQI).")
    return True


def main():
    print("\n" + "#" * 75)
    print("#  PEARLS AQI PREDICTOR - PRE-SUBMISSION AUDIT SUITE")
    print("#" * 75)

    audits = [
        ("Data Artifacts", audit_data_artifacts),
        ("Model Artifacts", audit_model_artifacts),
        ("Interpretability Plot", audit_interpretability),
        ("CI/CD Infrastructure", audit_cicd_infrastructure),
        ("Headless Inference", audit_dashboard_headless_inference)
    ]

    report = []
    all_passed = True

    for name, audit_fn in audits:
        try:
            success = audit_fn()
            report.append((name, "PASSED", "✓"))
        except Exception as e:
            print(f"  [FAIL] Audit '{name}' failed: {e}")
            report.append((name, f"FAILED: {e}", "✗"))
            all_passed = False

    print_banner("Audit Summary Table")
    print(f"  {'Component':30} | {'Status':12} | Check")
    print("  " + "-" * 55)
    for name, status, icon in report:
        print(f"  {name:30} | {status:12} | [{icon}]")
    print("=" * 75)

    if all_passed:
        print("  ALL AUDIT CHECKS PASSED PERFECTLY! DELIVERABLES VERIFIED FOR SUBMISSION.")
        print("=" * 75 + "\n")
        sys.exit(0)
    else:
        print("  AUDIT FAILED ON ONE OR MORE REQUIRED DELIVERABLES.")
        print("=" * 75 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
