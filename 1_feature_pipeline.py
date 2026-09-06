#!/usr/bin/env python3
"""
Pearls AQI Predictor - Feature Pipeline
Extracts air quality and weather data from Open-Meteo API, performs feature engineering,
and registers features with Hopsworks Feature Store or falls back cleanly to local persistent storage.
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("feature_pipeline")

# Load environment variables
load_dotenv()

# Built-in city coordinates
DEFAULT_CITIES = {
    "Karachi": {"lat": 24.8607, "lon": 67.0011, "country": "Pakistan"},
    "Lahore": {"lat": 31.5497, "lon": 74.3436, "country": "Pakistan"},
    "Islamabad": {"lat": 33.6844, "lon": 73.0479, "country": "Pakistan"}
}


def fetch_open_meteo_data(lat: float, lon: float, past_days: int = 90, forecast_days: int = 3) -> pd.DataFrame:
    """
    Fetches air quality and meteorological forecast & historical data from Open-Meteo.
    Zero-friction: requires no API keys.
    """
    logger.info(f"Fetching Open-Meteo Air Quality data (lat={lat}, lon={lon}, past_days={past_days}, forecast_days={forecast_days})...")
    
    # 1. Air Quality API
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "UTC"
    }
    
    try:
        aq_resp = requests.get(aq_url, params=aq_params, timeout=30)
        aq_resp.raise_for_status()
        aq_data = aq_resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch air quality data from Open-Meteo: {e}")
        raise

    hourly_aq = aq_data.get("hourly", {})
    if not hourly_aq or "time" not in hourly_aq:
        raise ValueError("Invalid air quality response structure from Open-Meteo API.")

    df_aq = pd.DataFrame(hourly_aq)
    df_aq["time"] = pd.to_datetime(df_aq["time"], utc=True)

    # 2. Weather Forecast API
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "UTC"
    }
    
    logger.info("Fetching Open-Meteo Weather data...")
    try:
        weather_resp = requests.get(weather_url, params=weather_params, timeout=30)
        weather_resp.raise_for_status()
        weather_data = weather_resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch weather data from Open-Meteo: {e}")
        raise

    hourly_weather = weather_data.get("hourly", {})
    df_weather = pd.DataFrame(hourly_weather)
    df_weather["time"] = pd.to_datetime(df_weather["time"], utc=True)

    # Merge on timestamp
    df = pd.merge(df_aq, df_weather, on="time", how="inner")
    df = df.sort_values("time").reset_index(drop=True)
    logger.info(f"Successfully extracted {len(df)} total hourly records.")
    return df


def engineer_features(df: pd.DataFrame, city_name: str, lat: float, lon: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans raw time series data, handles missing values, and derives temporal,
    lag, rolling, and meteorological features. Returns (historical_features, forecast_features).
    """
    logger.info("Engineering temporal, lag, rolling, and meteorological features...")
    df = df.copy()
    
    # Metadata tags
    df["city"] = city_name
    df["latitude"] = lat
    df["longitude"] = lon
    
    # Sort chronologically
    df = df.sort_values("time").reset_index(drop=True)

    # Impute missing raw sensor values (forward fill then backward fill)
    raw_pollutants = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "us_aqi"]
    raw_weather = ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "wind_direction_10m", "precipitation"]
    for col in raw_pollutants + raw_weather:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    # Temporal & Cyclical Features
    hours = df["time"].dt.hour
    dayofweek = df["time"].dt.dayofweek
    dayofyear = df["time"].dt.dayofyear
    month = df["time"].dt.month

    df["hour"] = hours
    df["dayofweek"] = dayofweek
    df["month"] = month
    df["is_weekend"] = (dayofweek >= 5).astype(int)

    df["sin_hour"] = np.sin(2 * np.pi * hours / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * hours / 24.0)
    df["sin_dayofweek"] = np.sin(2 * np.pi * dayofweek / 7.0)
    df["cos_dayofweek"] = np.cos(2 * np.pi * dayofweek / 7.0)
    df["sin_month"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * (month - 1) / 12.0)

    # Meteorological Wind Vectors (U and V components)
    rad = np.radians(df["wind_direction_10m"])
    df["wind_u"] = -df["wind_speed_10m"] * np.sin(rad)
    df["wind_v"] = -df["wind_speed_10m"] * np.cos(rad)

    # Lags and Rolling Windows for Target & Key Pollutants
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"lag_{lag}h_us_aqi"] = df["us_aqi"].shift(lag)
        df[f"lag_{lag}h_pm2_5"] = df["pm2_5"].shift(lag)

    for window in [6, 12, 24]:
        df[f"rolling_mean_{window}h_us_aqi"] = df["us_aqi"].rolling(window=window, min_periods=1).mean()
        df[f"rolling_std_{window}h_us_aqi"] = df["us_aqi"].rolling(window=window, min_periods=1).std().fillna(0)
        df[f"rolling_mean_{window}h_pm2_5"] = df["pm2_5"].rolling(window=window, min_periods=1).mean()

    # Fill initial lag NaNs using backward fill
    df = df.bfill()

    # Format timestamp representations for Hopsworks & Parquet compatibility
    df["timestamp_str"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["timestamp_ms"] = (df["time"].astype("int64") // 10**6).astype("int64")

    # Current UTC time for splitting historical vs future forecast horizon
    now_utc = datetime.now(timezone.utc)
    historical_df = df[df["time"] <= pd.Timestamp(now_utc)].copy().reset_index(drop=True)
    forecast_df = df[df["time"] > pd.Timestamp(now_utc)].copy().reset_index(drop=True)

    logger.info(f"Feature engineering complete. Historical samples: {len(historical_df)}, Future forecast samples: {len(forecast_df)}.")
    return historical_df, forecast_df


def save_to_hopsworks(historical_df: pd.DataFrame, forecast_df: pd.DataFrame, api_key: str, project_name: str | None = None) -> bool:
    """
    Attempts to save features to Hopsworks Feature Store.
    Returns True if successful, False otherwise.
    """
    try:
        import hopsworks
        logger.info("Hopsworks library found. Authenticating with Hopsworks Feature Store...")
        
        login_kwargs = {"api_key_value": api_key}
        if project_name:
            login_kwargs["project"] = project_name
            
        project = hopsworks.login(**login_kwargs)
        fs = project.get_feature_store()
        
        # Prepare DataFrame for Hopsworks (remove datetime with timezone or convert to string)
        hw_df = historical_df.copy()
        hw_df["time"] = hw_df["timestamp_str"]
        
        feature_group = fs.get_or_create_feature_group(
            name="aqi_features",
            version=1,
            primary_key=["city", "timestamp_str"],
            event_time="timestamp_ms",
            description="Hourly air quality and weather features for Pearls AQI Predictor",
            online_enabled=False
        )
        
        logger.info("Inserting historical features into Hopsworks Feature Group 'aqi_features'...")
        feature_group.insert(hw_df, write_options={"wait_for_job": False})
        logger.info("Successfully registered features in Hopsworks Feature Store!")
        return True
    except Exception as e:
        logger.warning(f"Hopsworks integration encounter: {e}")
        return False


def save_to_local(historical_df: pd.DataFrame, forecast_df: pd.DataFrame, data_dir: str = "data"):
    """
    Saves feature tables locally in Parquet format.
    """
    os.makedirs(data_dir, exist_ok=True)
    hist_path = os.path.join(data_dir, "aqi_features.parquet")
    forecast_path = os.path.join(data_dir, "aqi_forecast_features.parquet")
    meta_path = os.path.join(data_dir, "metadata.json")

    logger.info(f"[LOCAL STORAGE] Saving historical features to {hist_path} and latest_features.parquet...")
    historical_df.to_parquet(hist_path, index=False)
    latest_path = os.path.join(data_dir, "latest_features.parquet")
    historical_df.to_parquet(latest_path, index=False)
    
    logger.info(f"[LOCAL STORAGE] Saving forecast features to {forecast_path}...")
    forecast_df.to_parquet(forecast_path, index=False)

    metadata = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "historical_records": len(historical_df),
        "forecast_records": len(forecast_df),
        "columns": list(historical_df.columns),
        "city": historical_df["city"].iloc[0] if len(historical_df) > 0 else "Unknown",
        "latitude": float(historical_df["latitude"].iloc[0]) if len(historical_df) > 0 else 0.0,
        "longitude": float(historical_df["longitude"].iloc[0]) if len(historical_df) > 0 else 0.0,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"[LOCAL STORAGE] Metadata saved to {meta_path}")


def run_feature_pipeline(city: str = "Karachi", lat: float | None = None, lon: float | None = None,
                         past_days: int = 90, forecast_days: int = 3, data_dir: str = "data"):
    """
    Orchestrates extraction, transformation, feature engineering, and feature store persistence with local fallback.
    """
    if city in DEFAULT_CITIES and (lat is None or lon is None):
        lat = DEFAULT_CITIES[city]["lat"]
        lon = DEFAULT_CITIES[city]["lon"]
    elif lat is None or lon is None:
        lat = DEFAULT_CITIES["Karachi"]["lat"]
        lon = DEFAULT_CITIES["Karachi"]["lon"]
        city = "Karachi"

    logger.info(f"Starting Pearls AQI Feature Pipeline for {city} ({lat}, {lon})...")
    
    # 1. Extraction
    df_raw = fetch_open_meteo_data(lat=lat, lon=lon, past_days=past_days, forecast_days=forecast_days)
    
    # 2. Transformation & Feature Engineering
    hist_df, forecast_df = engineer_features(df_raw, city_name=city, lat=lat, lon=lon)
    
    # Always guarantee local persistence
    save_to_local(hist_df, forecast_df, data_dir=data_dir)
    
    # 3. Hopsworks Feature Store (if key is present)
    api_key = os.getenv("HOPSWORKS_API_KEY", "").strip()
    project_name = os.getenv("HOPSWORKS_PROJECT_NAME", "").strip() or None
    
    if api_key:
        logger.info("Found HOPSWORKS_API_KEY in environment. Attempting Feature Store synchronization...")
        success = save_to_hopsworks(hist_df, forecast_df, api_key=api_key, project_name=project_name)
        if not success:
            logger.info("[FALLBACK] Hopsworks upload was bypassed or failed. Local storage serves as the active feature store.")
    else:
        logger.info("[FALLBACK] HOPSWORKS_API_KEY environment variable not set. Using local persistent storage in data/.")

    # 4. Optional AQICN Live Ground-Truth Check
    try:
        from api import fetch_live_aqicn, get_aqicn_token
        aqicn_tok = get_aqicn_token()
        if aqicn_tok:
            logger.info("Found AQICN API token in environment. Querying reference ground station...")
            live_reading = fetch_live_aqicn(city=city, lat=lat, lon=lon)
            if live_reading.get("status") == "success" and live_reading.get("data"):
                logger.info(f"AQICN Reference Station: '{live_reading['data']['station_name']}' reported Live AQI: {live_reading['data']['aqi']}")
    except Exception as e:
        logger.debug(f"AQICN integration check skipped: {e}")

    logger.info("Pearls AQI Feature Pipeline completed successfully!")
    return hist_df, forecast_df


def main():
    parser = argparse.ArgumentParser(description="Pearls AQI Feature Pipeline")
    parser.add_argument("--city", type=str, default="Karachi", help="City name")
    parser.add_argument("--lat", type=float, default=None, help="Latitude")
    parser.add_argument("--lon", type=float, default=None, help="Longitude")
    parser.add_argument("--past-days", type=int, default=90, help="Days of historical backfill")
    parser.add_argument("--forecast-days", type=int, default=3, help="Days of future forecast")
    parser.add_argument("--data-dir", type=str, default="data", help="Output directory for local fallback")

    args = parser.parse_args()
    run_feature_pipeline(
        city=args.city,
        lat=args.lat,
        lon=args.lon,
        past_days=args.past_days,
        forecast_days=args.forecast_days,
        data_dir=args.data_dir
    )


if __name__ == "__main__":
    main()

