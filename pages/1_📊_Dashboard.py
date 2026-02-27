"""
Dashboard Page – Weather & Energy Visualization

Displays interactive charts showing:
    - Temperature trends over time (line chart)
    - Energy consumption patterns (line chart with dual axis)
    - Weather vs energy scatter plot
    - City comparison bar chart
    - Live weather forecast (from Open-Meteo API)

Sidebar controls allow filtering by city and date range.
"""

import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.database import get_records_filtered, get_cities, get_date_range, init_db
from utils.api_client import fetch_weather_forecast
from utils.charts import (
    line_chart, dual_axis_line_chart, scatter_chart, bar_chart, COLORS
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="📊 Dashboard – Weather & Energy",
    page_icon="📊",
    layout="wide",
)

st.markdown("# 📊 Weather & Energy Dashboard")
st.markdown("Explore how weather conditions influence energy consumption across Swiss cities.")

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

st.sidebar.markdown("## 🎛️ Dashboard Filters")

# City selector
cities_df = get_cities()
city_names = ["All"] + cities_df["name"].tolist() if not cities_df.empty else ["All"]
selected_city = st.sidebar.selectbox("🏙️ Select City", city_names, index=0)

# Date range selector
date_range = get_date_range()
if date_range[0]:
    min_date = pd.to_datetime(date_range[0]).date()
    max_date = pd.to_datetime(date_range[1]).date()
else:
    min_date = (datetime.today() - timedelta(days=365)).date()
    max_date = datetime.today().date()

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("📅 From", value=min_date, min_value=min_date, max_value=max_date)
with col2:
    end_date = st.date_input("📅 To", value=max_date, min_value=min_date, max_value=max_date)

# Chart type preference
chart_detail = st.sidebar.radio(
    "📊 Chart Detail",
    ["Daily", "Weekly Average", "Monthly Average"],
    index=0,
)

st.sidebar.divider()

# Forecast options
st.sidebar.markdown("### 🌤️ Forecast Settings")
forecast_city = st.sidebar.selectbox(
    "Forecast City",
    cities_df["name"].tolist() if not cities_df.empty else ["St. Gallen"],
    index=0,
    key="forecast_city",
)


# ---------------------------------------------------------------------------
# Load data based on filters
# ---------------------------------------------------------------------------

df = get_records_filtered(
    city=selected_city,
    start_date=str(start_date),
    end_date=str(end_date),
)

if df.empty:
    st.warning("⚠️ No data available for the selected filters. Try adjusting the date range or city.")
    st.stop()

# Ensure date is datetime type
df["date"] = pd.to_datetime(df["date"])

# ---------------------------------------------------------------------------
# Aggregate data based on chart detail selection
# ---------------------------------------------------------------------------

if chart_detail == "Weekly Average":
    df_agg = df.copy()
    df_agg["date"] = df_agg["date"].dt.to_period("W").apply(lambda r: r.start_time)
    df_agg = df_agg.groupby(["date", "city"]).agg({
        "temperature_avg": "mean",
        "temperature_min": "min",
        "temperature_max": "max",
        "humidity": "mean",
        "wind_speed": "mean",
        "precipitation": "sum",
        "energy_kwh": "mean",
    }).reset_index()
elif chart_detail == "Monthly Average":
    df_agg = df.copy()
    df_agg["date"] = df_agg["date"].dt.to_period("M").apply(lambda r: r.start_time)
    df_agg = df_agg.groupby(["date", "city"]).agg({
        "temperature_avg": "mean",
        "temperature_min": "min",
        "temperature_max": "max",
        "humidity": "mean",
        "wind_speed": "mean",
        "precipitation": "sum",
        "energy_kwh": "mean",
    }).reset_index()
else:
    df_agg = df.copy()


# ---------------------------------------------------------------------------
# Charts Section
# ---------------------------------------------------------------------------

# --- 1. Temperature & Energy dual-axis chart ---
st.subheader("🌡️ Temperature vs Energy Consumption")

# Average across cities for the dual-axis chart
daily_combined = df_agg.groupby("date").agg({
    "temperature_avg": "mean",
    "energy_kwh": "mean",
}).reset_index()

fig_dual = dual_axis_line_chart(
    daily_combined, x="date",
    y1="temperature_avg", y2="energy_kwh",
    title="Temperature & Energy Consumption Over Time",
    y1_label="Temperature (°C)", y2_label="Energy (kWh)",
)
st.plotly_chart(fig_dual, width="stretch")

# --- 2. Energy consumption by city ---
st.subheader("⚡ Energy Consumption by City")
col1, col2 = st.columns(2)

with col1:
    # Line chart: energy over time per city
    fig_energy = line_chart(
        df_agg, x="date", y="energy_kwh",
        color="city",
        title="Daily Energy Consumption per City",
        y_label="Energy (kWh)",
    )
    st.plotly_chart(fig_energy, width="stretch")

with col2:
    # Bar chart: average energy by city
    city_avg = df.groupby("city")["energy_kwh"].mean().reset_index()
    city_avg.columns = ["city", "avg_energy_kwh"]
    fig_bar = bar_chart(
        city_avg, x="city", y="avg_energy_kwh",
        title="Average Daily Energy by City",
    )
    st.plotly_chart(fig_bar, width="stretch")


# --- 3. Weather scatter plot ---
st.subheader("🔗 Temperature vs Energy Correlation")

# User selects which weather feature to plot against energy
weather_feature = st.selectbox(
    "Select weather feature for scatter plot:",
    ["temperature_avg", "humidity", "wind_speed", "precipitation"],
    format_func=lambda x: {
        "temperature_avg": "🌡️ Average Temperature (°C)",
        "humidity": "💧 Humidity (%)",
        "wind_speed": "💨 Wind Speed (km/h)",
        "precipitation": "🌧️ Precipitation (mm)",
    }[x],
)

fig_scatter = scatter_chart(
    df, x=weather_feature, y="energy_kwh",
    color="city" if selected_city == "All" else None,
    title=f"Energy Consumption vs {weather_feature.replace('_', ' ').title()}",
    trendline="ols",
)
st.plotly_chart(fig_scatter, width="stretch")


# --- 4. Weather Forecast ---
st.divider()
st.subheader(f"🌤️ 7-Day Weather Forecast – {forecast_city}")

# Get city coordinates for API call
city_row = cities_df[cities_df["name"] == forecast_city]
if not city_row.empty:
    lat = city_row.iloc[0]["latitude"]
    lon = city_row.iloc[0]["longitude"]

    with st.spinner("Fetching forecast from Open-Meteo API..."):
        forecast_df = fetch_weather_forecast(lat, lon, days=7)

    if not forecast_df.empty:
        # Display forecast as a clean table
        forecast_display = forecast_df.copy()
        forecast_display.columns = [
            "Date", "Avg Temp (°C)", "Min Temp (°C)", "Max Temp (°C)",
            "Humidity (%)", "Wind (km/h)", "Rain (mm)"
        ]
        forecast_display["Date"] = forecast_display["Date"].dt.strftime("%a, %b %d")
        st.dataframe(forecast_display, width="stretch", hide_index=True)

        # Forecast chart
        fig_forecast = line_chart(
            forecast_df, x="date", y=["temperature_avg", "temperature_min", "temperature_max"],
            title="7-Day Temperature Forecast",
            y_label="Temperature (°C)",
        )
        st.plotly_chart(fig_forecast, width="stretch")
    else:
        st.info("📡 Could not fetch forecast data. The app is using cached data for other features.")
else:
    st.warning("City coordinates not found.")
