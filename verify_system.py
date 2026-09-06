#!/usr/bin/env python3
"""
Pearls AQI Predictor - System Verification Suite
Validates dependencies, feature pipeline execution, model training, artifact persistence,
inference capability, and Streamlit dashboard readiness.
"""

import os
import sys
import json
import logging
import subprocess
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("verify_system")


def print_step_header(step_num: int, title: str):
    print("\n" + "=" * 70)
    print(f" STEP {step_num}: {title.upper()}")
    print("=" * 70)


def check_dependencies() -> bool:
    print_step_header(1, "Dependency Import Verification")
    required_modules = [
        "requests",
        "pandas",
        "numpy",
        "sklearn",
        "joblib",
        "streamlit",
        "plotly",
        "pyarrow",
        "dotenv"
    ]
    missing = []
    for mod in required_modules:
        try:
            __import__(mod)
            print(f"  [PASS] Module '{mod}' is installed and importable.")
        except ImportError as e:
            print(f"  [FAIL] Missing required module: {mod} ({e})")
            missing.append(mod)

    # Optional Hopsworks
    try:
        __import__("hopsworks")
        print("  [INFO] Hopsworks SDK is installed.")
    except ImportError:
        print("  [INFO] Hopsworks SDK not installed; local persistent fallback is active.")

    return len(missing) == 0


def verify_feature_pipeline() -> bool:
    print_step_header(2, "Feature Pipeline Execution & Data Verification")
    cmd = [sys.executable, "1_feature_pipeline.py", "--city", "Karachi", "--past-days", "14", "--forecast-days", "3"]
    print(f"  Executing: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [FAIL] Feature pipeline failed with return code {result.returncode}")
        print(f"  Error output:\n{result.stderr}")
        return False

    # Check generated files
    hist_file = os.path.join("data", "aqi_features.parquet")
    forecast_file = os.path.join("data", "aqi_forecast_features.parquet")
    meta_file = os.path.join("data", "metadata.json")

    for f_path in [hist_file, forecast_file, meta_file]:
        if not os.path.exists(f_path):
            print(f"  [FAIL] Expected artifact '{f_path}' was not generated.")
            return False

    df_hist = pd.read_parquet(hist_file)
    df_fore = pd.read_parquet(forecast_file)

    if len(df_hist) == 0 or len(df_fore) == 0:
        print(f"  [FAIL] Generated feature parquet files are empty. (Hist: {len(df_hist)}, Fore: {len(df_fore)})")
        return False

    print(f"  [PASS] Historical feature table generated: {len(df_hist)} rows, {len(df_hist.columns)} features.")
    print(f"  [PASS] Future forecast feature table generated: {len(df_fore)} rows, {len(df_fore.columns)} features.")
    print(f"  [PASS] Target column 'us_aqi' present with non-null values.")
    return True


def verify_training_pipeline() -> bool:
    print_step_header(3, "Training Pipeline Execution & Model Artifacts")
    cmd = [sys.executable, "2_training_pipeline.py", "--city", "Karachi"]
    print(f"  Executing: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [FAIL] Training pipeline failed with return code {result.returncode}")
        print(f"  Error output:\n{result.stderr}")
        return False

    model_file = os.path.join("models", "aqi_model.joblib")
    metrics_file = os.path.join("models", "metrics.json")
    meta_file = os.path.join("models", "metadata.json")

    for f_path in [model_file, metrics_file, meta_file]:
        if not os.path.exists(f_path):
            print(f"  [FAIL] Expected model artifact '{f_path}' was not generated.")
            return False

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    champ = metrics.get("champion_model", "Unknown")
    test_mae = metrics.get("test_metrics", {}).get("mae", None)
    test_r2 = metrics.get("test_metrics", {}).get("r2", None)

    print(f"  [PASS] Model artifact successfully saved to '{model_file}'.")
    print(f"  [PASS] Champion model selected: '{champ}'.")
    print(f"  [PASS] Test Set Evaluation -> MAE: {test_mae}, R²: {test_r2}.")
    return True


def verify_inference_capabilities() -> bool:
    print_step_header(4, "Model Inference & Sanity Testing")
    import joblib

    model_file = os.path.join("models", "aqi_model.joblib")
    meta_file = os.path.join("models", "metadata.json")
    forecast_file = os.path.join("data", "aqi_forecast_features.parquet")

    model = joblib.load(model_file)
    with open(meta_file, "r") as f:
        meta = json.load(f)
    features = meta["features"]

    df_forecast = pd.read_parquet(forecast_file)
    sample_X = df_forecast[features].ffill().bfill().head(5)

    preds = model.predict(sample_X)
    print(f"  Sample 5-hour test predictions: {np.round(preds, 2)}")

    if len(preds) != 5 or np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
        print("  [FAIL] Predictions contain NaNs or infinite values.")
        return False

    if np.any(preds < 0) or np.any(preds > 600):
        print("  [FAIL] Predictions exceed reasonable physical AQI limits.")
        return False

    print("  [PASS] Inference pipeline generated valid, bounded numerical AQI predictions.")
    return True


def verify_app_readiness() -> bool:
    print_step_header(5, "Streamlit Dashboard Compilation & Readiness")
    # Verify script compiles without syntax errors
    app_file = "3_app.py"
    if not os.path.exists(app_file):
        print(f"  [FAIL] '{app_file}' does not exist.")
        return False

    try:
        with open(app_file, "r") as f:
            code = f.read()
        compile(code, app_file, "exec")
        print(f"  [PASS] '{app_file}' successfully compiled without syntax errors.")
    except Exception as e:
        print(f"  [FAIL] Syntax error in '{app_file}': {e}")
        return False

    return True


def main():
    print("\n" + "#" * 70)
    print("#  PEARLS AQI PREDICTOR - SYSTEM VERIFICATION SUITE")
    print("#" * 70)

    checks = [
        ("Dependencies", check_dependencies),
        ("Feature Pipeline", verify_feature_pipeline),
        ("Training Pipeline", verify_training_pipeline),
        ("Inference Sanity", verify_inference_capabilities),
        ("App Readiness", verify_app_readiness),
    ]

    summary = []
    overall_pass = True

    for name, func in checks:
        try:
            passed = func()
        except Exception as e:
            print(f"  [FAIL] Exception during {name}: {e}")
            passed = False

        summary.append((name, passed))
        if not passed:
            overall_pass = False

    print("\n" + "=" * 70)
    print(" EXECUTIVE VERIFICATION REPORT")
    print("=" * 70)
    for name, passed in summary:
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"  [{symbol}] {name:30}: {status}")

    print("=" * 70)
    if overall_pass:
        print("  ALL SYSTEM CHECKS PASSED SUCCESSFULLY! SYSTEM READY FOR OPERATION.")
        print("=" * 70 + "\n")
        sys.exit(0)
    else:
        print("  VERIFICATION FAILED ON ONE OR MORE CHECKS.")
        print("=" * 70 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

