"""
API client for the Open-Meteo weather service.

Open-Meteo provides free weather data without requiring an API key.
This module handles:
- Fetching historical weather data (archive API)
- Fetching weather forecasts (forecast API)
- Graceful fallback to cached/sample data when the API is unavailable

Endpoints used:
  - https://archive-api.open-meteo.com/v1/archive  (historical data)
  - https://api.open-meteo.com/v1/forecast          (7-day forecast)
"""

import requests
import json
import os
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Base URLs for Open-Meteo APIs
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Path to cached sample data for offline fallback
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CACHE_FILE = os.path.join(CACHE_DIR, "sample_weather.json")

# Request timeout in seconds
TIMEOUT = 10


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------

def fetch_weather_history(latitude, longitude, start_date, end_date):
    """
    Fetch historical daily weather data from the Open-Meteo Archive API.

    Args:
        latitude (float): Location latitude (e.g., 47.42 for St. Gallen)
        longitude (float): Location longitude (e.g., 9.37 for St. Gallen)
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format

    Returns:
        pd.DataFrame: Daily weather data with columns:
            [date, temperature_avg, temperature_min, temperature_max,
             humidity, wind_speed, precipitation]

    Falls back to cached data if the API is unreachable.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join([
            "temperature_2m_mean",
            "temperature_2m_min",
            "temperature_2m_max",
            "relative_humidity_2m_mean",
            "wind_speed_10m_max",
            "precipitation_sum",
        ]),
        "timezone": "Europe/Zurich",
    }

    try:
        response = requests.get(ARCHIVE_URL, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()

        # Cache the response for offline use
        _save_cache(data, f"history_{latitude}_{longitude}")

        return _parse_daily_data(data)

    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"⚠️ API call failed ({e}). Using cached data.")
        return _load_cached_data("history")


def fetch_weather_forecast(latitude, longitude, days=7):
    """
    Fetch weather forecast from the Open-Meteo Forecast API.

    Args:
        latitude (float): Location latitude
        longitude (float): Location longitude
        days (int): Number of forecast days (1-16, default 7)

    Returns:
        pd.DataFrame: Forecast data with same columns as historical data

    Falls back to cached data if the API is unreachable.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ",".join([
            "temperature_2m_mean",
            "temperature_2m_min",
            "temperature_2m_max",
            "relative_humidity_2m_mean",
            "wind_speed_10m_max",
            "precipitation_sum",
        ]),
        "timezone": "Europe/Zurich",
        "forecast_days": days,
    }

    try:
        response = requests.get(FORECAST_URL, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()

        # Cache the response for offline use
        _save_cache(data, f"forecast_{latitude}_{longitude}")

        return _parse_daily_data(data)

    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"⚠️ Forecast API call failed ({e}). Using cached data.")
        return _load_cached_data("forecast")


# ---------------------------------------------------------------------------
# Internal helper functions
# ---------------------------------------------------------------------------

def _parse_daily_data(api_response):
    """
    Parse the Open-Meteo API JSON response into a clean DataFrame.

    The API returns data in a nested structure under 'daily'.
    We rename columns to our internal naming convention.

    Args:
        api_response (dict): Raw JSON response from Open-Meteo

    Returns:
        pd.DataFrame: Cleaned weather data
    """
    daily = api_response.get("daily", {})

    df = pd.DataFrame({
        "date": daily.get("time", []),
        "temperature_avg": daily.get("temperature_2m_mean", []),
        "temperature_min": daily.get("temperature_2m_min", []),
        "temperature_max": daily.get("temperature_2m_max", []),
        "humidity": daily.get("relative_humidity_2m_mean", []),
        "wind_speed": daily.get("wind_speed_10m_max", []),
        "precipitation": daily.get("precipitation_sum", []),
    })

    # Convert date strings to datetime objects
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])

    return df


def _save_cache(data, prefix):
    """
    Save API response to a JSON file for offline fallback.

    Args:
        data (dict): API response data
        prefix (str): Cache file prefix (e.g., 'history_47.42_9.37')
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"cache_{prefix}.json")
    try:
        with open(cache_path, "w") as f:
            json.dump(data, f)
    except IOError:
        pass  # Silently fail on cache write errors


def _load_cached_data(data_type):
    """
    Load cached weather data from disk. If no cache exists, returns
    a synthetic sample dataset so the app still works offline.

    Args:
        data_type (str): 'history' or 'forecast'

    Returns:
        pd.DataFrame: Weather data (cached or synthetic fallback)
    """
    # Try to find any cached file
    if os.path.exists(CACHE_DIR):
        for filename in os.listdir(CACHE_DIR):
            if filename.startswith(f"cache_{data_type}") and filename.endswith(".json"):
                try:
                    with open(os.path.join(CACHE_DIR, filename), "r") as f:
                        data = json.load(f)
                    return _parse_daily_data(data)
                except (IOError, json.JSONDecodeError):
                    continue

    # Ultimate fallback: generate synthetic data
    return _generate_synthetic_weather(data_type)


def _generate_synthetic_weather(data_type):
    """
    Generate synthetic weather data as a last-resort fallback.
    Uses realistic seasonal patterns for central Switzerland.

    Args:
        data_type (str): 'history' or 'forecast'

    Returns:
        pd.DataFrame: Synthetic weather data
    """
    import numpy as np

    if data_type == "forecast":
        # Generate 7 days of forecast data starting from today
        base_date = datetime.today()
        dates = [base_date + timedelta(days=i) for i in range(7)]
    else:
        # Generate 365 days of historical data
        base_date = datetime.today() - timedelta(days=365)
        dates = [base_date + timedelta(days=i) for i in range(365)]

    n = len(dates)
    np.random.seed(42)

    # Seasonal temperature pattern (cold winter, warm summer for CH)
    day_of_year = [(d.timetuple().tm_yday) for d in dates]
    temp_base = [10 + 12 * np.sin((doy - 80) * 2 * np.pi / 365) for doy in day_of_year]

    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "temperature_avg": [t + np.random.normal(0, 2) for t in temp_base],
        "temperature_min": [t - 3 + np.random.normal(0, 1.5) for t in temp_base],
        "temperature_max": [t + 4 + np.random.normal(0, 1.5) for t in temp_base],
        "humidity": np.clip(np.random.normal(70, 15, n), 30, 100),
        "wind_speed": np.clip(np.random.exponential(8, n), 0, 50),
        "precipitation": np.clip(np.random.exponential(2, n), 0, 40),
    })

    return df
