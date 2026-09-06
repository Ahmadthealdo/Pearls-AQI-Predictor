#!/usr/bin/env python3
"""
Pearls AQI Predictor - AQICN / WAQI API Integration
Fetches ground-truth real-time air quality observations from the World Air Quality Index (WAQI / AQICN) API.
Enables real-time comparison between ML model forecasts and physical monitoring station sensors.
"""

import os
import sys
import logging
import argparse
import json
import requests
from dotenv import load_dotenv

# Load environment variables at the very top
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("aqicn_api")

# Known token environment variable keys in order of precedence
TOKEN_KEYS = ["AQICN_API_TOKEN", "AQICN_TOKEN", "WAQI_API_TOKEN", "WAQI_TOKEN"]


def get_aqicn_token() -> str | None:
    """Retrieves the AQICN / WAQI API token from environment variables (.env)."""
    for key in TOKEN_KEYS:
        token = os.getenv(key, "").strip()
        if token:
            return token
    return None


def fetch_live_aqicn(city: str = "Karachi", lat: float | None = None, lon: float | None = None,
                     token: str | None = None) -> dict:
    """
    Fetches live ground station air quality telemetry from the AQICN/WAQI API.
    Zero-crash guarantee: returns structured status dict even on network or auth errors.
    """
    api_token = token or get_aqicn_token()
    
    if not api_token:
        logger.warning("No AQICN API token found in environment. Set AQICN_API_TOKEN in .env.")
        return {
            "status": "missing_token",
            "message": "AQICN_API_TOKEN is not set in .env. Real-time ground station comparison is disabled.",
            "data": None
        }

    # Construct endpoint URL
    if lat is not None and lon is not None:
        url = f"https://api.waqi.info/feed/geo:{lat};{lon}/"
    else:
        # Sanitize city query
        clean_city = city.lower().replace(" ", "-")
        url = f"https://api.waqi.info/feed/{clean_city}/"

    params = {"token": api_token}

    logger.info(f"Querying AQICN API endpoint for {city} (geo: {lat}, {lon})...")
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        res_json = resp.json()

        if res_json.get("status") != "ok":
            msg = res_json.get("data", "Unknown AQICN API error")
            logger.warning(f"AQICN API returned error status: {msg}")
            return {
                "status": "api_error",
                "message": f"AQICN API responded with status '{res_json.get('status')}': {msg}",
                "data": None
            }

        raw_data = res_json.get("data", {})
        iaqi = raw_data.get("iaqi", {})
        city_meta = raw_data.get("city", {})
        time_meta = raw_data.get("time", {})

        # Extract parsed telemetry
        parsed = {
            "aqi": float(raw_data.get("aqi", 0)),
            "dominant_pollutant": raw_data.get("dominentpol", "unknown"),
            "station_name": city_meta.get("name", city),
            "station_url": city_meta.get("url", ""),
            "coordinates": city_meta.get("geo", [lat, lon]),
            "timestamp": time_meta.get("s", ""),
            "timezone": time_meta.get("tz", ""),
            "pollutants": {
                "pm25": iaqi.get("pm25", {}).get("v"),
                "pm10": iaqi.get("pm10", {}).get("v"),
                "no2": iaqi.get("no2", {}).get("v"),
                "o3": iaqi.get("o3", {}).get("v"),
                "so2": iaqi.get("so2", {}).get("v"),
                "co": iaqi.get("co", {}).get("v"),
            },
            "weather": {
                "temperature": iaqi.get("t", {}).get("v"),
                "humidity": iaqi.get("h", {}).get("v"),
                "pressure": iaqi.get("p", {}).get("v"),
                "wind": iaqi.get("w", {}).get("v"),
            },
            "attributions": raw_data.get("attributions", [])
        }

        logger.info(f"Successfully retrieved live AQICN data for '{parsed['station_name']}' (AQI: {parsed['aqi']}).")
        return {
            "status": "success",
            "message": "Live ground station telemetry retrieved successfully.",
            "data": parsed
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"AQICN network request failed: {e}")
        return {
            "status": "network_error",
            "message": f"Network communication error with AQICN: {str(e)}",
            "data": None
        }
    except Exception as e:
        logger.error(f"Unexpected error parsing AQICN response: {e}")
        return {
            "status": "parse_error",
            "message": f"Error parsing AQICN response: {str(e)}",
            "data": None
        }


def compare_prediction_with_live(predicted_aqi: float, live_data: dict | None) -> dict:
    """
    Computes delta and alignment analysis between predicted AQI and live AQICN sensor ground truth.
    """
    if not live_data:
        return {
            "has_comparison": False,
            "reason": "Live AQICN data not available."
        }

    live_aqi = live_data.get("aqi", 0.0)
    delta = round(predicted_aqi - live_aqi, 2)
    abs_err = round(abs(delta), 2)
    pct_err = round((abs_err / max(live_aqi, 1.0)) * 100, 1)

    if abs_err <= 15.0:
        alignment = "Excellent Alignment"
        color = "#10B981"
    elif abs_err <= 30.0:
        alignment = "Good Alignment"
        color = "#3B82F6"
    elif delta > 0:
        alignment = "Model Overestimating"
        color = "#F59E0B"
    else:
        alignment = "Model Underestimating"
        color = "#EF4444"

    return {
        "has_comparison": True,
        "predicted_aqi": round(predicted_aqi, 1),
        "live_aqi": round(live_aqi, 1),
        "delta": delta,
        "absolute_error": abs_err,
        "percentage_error": pct_err,
        "alignment": alignment,
        "color": color,
        "station_name": live_data.get("station_name", "Unknown Station"),
        "dominant_pollutant": live_data.get("dominant_pollutant", "unknown"),
        "timestamp": live_data.get("timestamp", "")
    }


def main():
    parser = argparse.ArgumentParser(description="AQICN Live Telemetry Tool")
    parser.add_argument("--city", type=str, default="Karachi", help="City name")
    parser.add_argument("--lat", type=float, default=None, help="Latitude")
    parser.add_argument("--lon", type=float, default=None, help="Longitude")
    parser.add_argument("--token", type=str, default=None, help="Explicit AQICN token")
    args = parser.parse_args()

    result = fetch_live_aqicn(city=args.city, lat=args.lat, lon=args.lon, token=args.token)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

