"""
Add Data Page – Form for Adding New Energy Records

Enables users to:
    - Manually enter an energy consumption record with weather data
    - Auto-fill weather data from the Open-Meteo API for a selected city and date
    - View recently added records
    - Delete records (with confirmation)
"""

import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.database import add_record, get_all_records, get_cities, delete_record
from utils.api_client import fetch_weather_history

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="📝 Add Data",
    page_icon="📝",
    layout="wide",
)

st.markdown("# 📝 Add Energy Consumption Record")
st.markdown("Manually enter a new energy consumption record or auto-fill weather data from the API.")


# ---------------------------------------------------------------------------
# Sidebar: recent records summary
# ---------------------------------------------------------------------------

st.sidebar.markdown("## 📋 Quick Stats")
df_all = get_all_records()
st.sidebar.metric("Total Records", f"{len(df_all):,}")
if not df_all.empty:
    st.sidebar.metric("Latest Record", df_all["date"].max())
    st.sidebar.metric("Avg Energy", f"{df_all['energy_kwh'].mean():.1f} kWh")


# ---------------------------------------------------------------------------
# Auto-fill section
# ---------------------------------------------------------------------------

st.subheader("🌐 Auto-Fill Weather Data from API")
st.markdown("Select a city and date to fetch weather data automatically from Open-Meteo.")

cities_df = get_cities()
city_names = cities_df["name"].tolist() if not cities_df.empty else ["St. Gallen"]

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    autofill_city = st.selectbox("🏙️ City", city_names, key="autofill_city")

with col2:
    autofill_date = st.date_input(
        "📅 Date",
        value=datetime.today().date() - timedelta(days=1),
        max_value=datetime.today().date(),
        key="autofill_date",
    )

with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    autofill_clicked = st.button("📡 Fetch Weather", type="secondary")

# Store auto-filled values in session state
if "autofill_weather" not in st.session_state:
    st.session_state.autofill_weather = None

if autofill_clicked:
    city_row = cities_df[cities_df["name"] == autofill_city]
    if not city_row.empty:
        lat = city_row.iloc[0]["latitude"]
        lon = city_row.iloc[0]["longitude"]
        date_str = str(autofill_date)

        with st.spinner("Fetching weather data..."):
            weather_df = fetch_weather_history(lat, lon, date_str, date_str)

        if not weather_df.empty:
            row = weather_df.iloc[0]
            st.session_state.autofill_weather = {
                "temp_avg": float(row.get("temperature_avg", 10.0)),
                "temp_min": float(row.get("temperature_min", 5.0)),
                "temp_max": float(row.get("temperature_max", 15.0)),
                "humidity": float(row.get("humidity", 70.0)),
                "wind_speed": float(row.get("wind_speed", 10.0)),
                "precipitation": float(row.get("precipitation", 0.0)),
            }
            st.success(f"✅ Weather data fetched for {autofill_city} on {date_str}!")
        else:
            st.warning("⚠️ Could not fetch weather data. Enter values manually below.")

# Get default values (from auto-fill or generic defaults)
defaults = st.session_state.autofill_weather or {
    "temp_avg": 10.0, "temp_min": 5.0, "temp_max": 15.0,
    "humidity": 70.0, "wind_speed": 10.0, "precipitation": 0.0,
}


# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------

st.divider()
st.subheader("📋 Record Details")

with st.form("add_record_form", clear_on_submit=True):
    col1, col2 = st.columns(2)

    with col1:
        form_date = st.date_input(
            "📅 Date*",
            value=autofill_date if autofill_clicked else datetime.today().date() - timedelta(days=1),
            max_value=datetime.today().date(),
        )
        form_city = st.selectbox("🏙️ City*", city_names)
        form_energy = st.number_input(
            "⚡ Energy Consumption (kWh)*",
            value=18.0, min_value=0.0, max_value=500.0, step=0.5,
            help="Daily energy consumption in kilowatt-hours",
        )

    with col2:
        form_temp_avg = st.number_input(
            "🌡️ Avg Temperature (°C)",
            value=defaults["temp_avg"], min_value=-30.0, max_value=50.0, step=0.1,
        )
        form_temp_min = st.number_input(
            "🌡️ Min Temperature (°C)",
            value=defaults["temp_min"], min_value=-40.0, max_value=45.0, step=0.1,
        )
        form_temp_max = st.number_input(
            "🌡️ Max Temperature (°C)",
            value=defaults["temp_max"], min_value=-20.0, max_value=55.0, step=0.1,
        )

    col3, col4, col5 = st.columns(3)

    with col3:
        form_humidity = st.number_input(
            "💧 Humidity (%)",
            value=defaults["humidity"], min_value=0.0, max_value=100.0, step=1.0,
        )
    with col4:
        form_wind = st.number_input(
            "💨 Wind Speed (km/h)",
            value=defaults["wind_speed"], min_value=0.0, max_value=200.0, step=0.5,
        )
    with col5:
        form_precip = st.number_input(
            "🌧️ Precipitation (mm)",
            value=defaults["precipitation"], min_value=0.0, max_value=200.0, step=0.5,
        )

    # Validation note
    st.caption("* Required fields. Weather values can be fetched automatically using the button above.")

    submitted = st.form_submit_button("💾 Save Record", type="primary", width="stretch")

    if submitted:
        # Input validation
        errors = []
        if form_temp_min > form_temp_max:
            errors.append("Min temperature cannot be greater than max temperature.")
        if form_temp_avg < form_temp_min or form_temp_avg > form_temp_max:
            errors.append("Average temperature should be between min and max.")
        if form_energy <= 0:
            errors.append("Energy consumption must be positive.")

        if errors:
            for error in errors:
                st.error(f"❌ {error}")
        else:
            # Attempt to insert into database
            success = add_record(
                date=str(form_date),
                city=form_city,
                temp_avg=round(form_temp_avg, 1),
                temp_min=round(form_temp_min, 1),
                temp_max=round(form_temp_max, 1),
                humidity=round(form_humidity, 1),
                wind_speed=round(form_wind, 1),
                precipitation=round(form_precip, 1),
                energy_kwh=round(form_energy, 2),
            )

            if success:
                st.success(f"✅ Record saved: {form_city} on {form_date} – {form_energy} kWh")
                # Clear auto-fill cache
                st.session_state.autofill_weather = None
            else:
                st.warning(f"⚠️ A record for {form_city} on {form_date} already exists. "
                           "Each city can only have one record per day.")


# ---------------------------------------------------------------------------
# Recent records
# ---------------------------------------------------------------------------

st.divider()
st.subheader("📋 Recent Records")

recent_df = get_all_records()
if not recent_df.empty:
    display_df = recent_df.head(20).copy()
    display_df = display_df[["id", "date", "city", "temperature_avg", "humidity",
                              "energy_kwh", "created_at"]]
    display_df.columns = ["ID", "Date", "City", "Temp (°C)", "Humidity (%)",
                           "Energy (kWh)", "Added At"]
    st.dataframe(display_df, width="stretch", hide_index=True)

    # Delete record option
    with st.expander("🗑️ Delete a Record"):
        delete_id = st.number_input(
            "Enter Record ID to delete",
            min_value=1, step=1, value=1,
        )
        if st.button("🗑️ Delete", type="secondary"):
            if delete_record(int(delete_id)):
                st.success(f"✅ Record #{delete_id} deleted successfully.")
                st.rerun()
            else:
                st.error(f"❌ Record #{delete_id} not found.")
else:
    st.info("No records in the database yet. Use the form above to add the first one!")
