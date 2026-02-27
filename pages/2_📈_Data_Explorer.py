"""
Data Explorer Page – Filter, Explore, and Analyze Data

Provides an interactive data exploration interface with:
    - Sidebar filters (city, date range, energy range)
    - Summary statistics cards
    - Full data table with sorting
    - Correlation heatmap
    - Distribution histograms
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.database import get_records_filtered, get_cities, get_date_range
from utils.charts import correlation_heatmap, bar_chart, line_chart, COLORS

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="📈 Data Explorer",
    page_icon="📈",
    layout="wide",
)

st.markdown("# 📈 Data Explorer")
st.markdown("Filter, explore, and analyze the energy consumption dataset.")

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

st.sidebar.markdown("## 🔍 Data Filters")

# City filter
cities_df = get_cities()
city_names = ["All"] + cities_df["name"].tolist() if not cities_df.empty else ["All"]
selected_city = st.sidebar.selectbox("🏙️ City", city_names, index=0)

# Date range
date_range = get_date_range()
if date_range[0]:
    min_date = pd.to_datetime(date_range[0]).date()
    max_date = pd.to_datetime(date_range[1]).date()
else:
    min_date = (datetime.today() - timedelta(days=365)).date()
    max_date = datetime.today().date()

start_date = st.sidebar.date_input("📅 Start Date", value=min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("📅 End Date", value=max_date, min_value=min_date, max_value=max_date)

# Energy range filter
st.sidebar.markdown("### ⚡ Energy Range (kWh)")
energy_min = st.sidebar.number_input("Min kWh", value=0.0, min_value=0.0, step=1.0)
energy_max = st.sidebar.number_input("Max kWh", value=100.0, min_value=0.0, step=1.0)

# Column selection
st.sidebar.markdown("### 📋 Display Columns")
all_columns = [
    "date", "city", "temperature_avg", "temperature_min", "temperature_max",
    "humidity", "wind_speed", "precipitation", "energy_kwh"
]
selected_columns = st.sidebar.multiselect(
    "Select columns to display",
    all_columns,
    default=all_columns,
)

# ---------------------------------------------------------------------------
# Load and filter data
# ---------------------------------------------------------------------------

df = get_records_filtered(
    city=selected_city,
    start_date=str(start_date),
    end_date=str(end_date),
)

# Apply energy range filter
if not df.empty:
    df = df[(df["energy_kwh"] >= energy_min) & (df["energy_kwh"] <= energy_max)]

if df.empty:
    st.warning("⚠️ No records match your filter criteria. Try adjusting the filters.")
    st.stop()

# Convert date column
df["date"] = pd.to_datetime(df["date"])


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

st.subheader("📊 Summary Statistics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("📋 Records", f"{len(df):,}")
with col2:
    st.metric("⚡ Avg Energy", f"{df['energy_kwh'].mean():.1f} kWh")
with col3:
    st.metric("🌡️ Avg Temp", f"{df['temperature_avg'].mean():.1f}°C")
with col4:
    st.metric("💧 Avg Humidity", f"{df['humidity'].mean():.0f}%")
with col5:
    st.metric("💨 Avg Wind", f"{df['wind_speed'].mean():.1f} km/h")


# ---------------------------------------------------------------------------
# Data table
# ---------------------------------------------------------------------------

st.subheader("📋 Data Table")

# Display selected columns
display_df = df[selected_columns].copy() if selected_columns else df.copy()

# Format date column for display if present
if "date" in display_df.columns:
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")

# Sort options
sort_col = st.selectbox(
    "Sort by",
    selected_columns if selected_columns else all_columns,
    index=0,
)
sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)

display_df = display_df.sort_values(
    sort_col,
    ascending=(sort_order == "Ascending"),
)

st.dataframe(
    display_df,
    width="stretch",
    hide_index=True,
    height=400,
)

# Download button
csv = display_df.to_csv(index=False)
st.download_button(
    label="📥 Download as CSV",
    data=csv,
    file_name="energy_data_export.csv",
    mime="text/csv",
)


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🔗 Correlation Analysis")

# Select numeric columns for correlation
numeric_cols = [
    "temperature_avg", "temperature_min", "temperature_max",
    "humidity", "wind_speed", "precipitation", "energy_kwh"
]
available_numeric = [c for c in numeric_cols if c in df.columns]

if len(available_numeric) >= 2:
    fig_corr = correlation_heatmap(df, columns=available_numeric)
    st.plotly_chart(fig_corr, width="stretch")

    # Highlight key correlations with energy
    st.markdown("**Key correlations with energy consumption:**")
    correlations = df[available_numeric].corr()["energy_kwh"].drop("energy_kwh").sort_values()

    for feature, corr_val in correlations.items():
        direction = "📈" if corr_val > 0 else "📉"
        strength = "Strong" if abs(corr_val) > 0.5 else "Moderate" if abs(corr_val) > 0.3 else "Weak"
        st.markdown(f"- {direction} **{feature}**: {corr_val:.3f} ({strength})")


# ---------------------------------------------------------------------------
# Distribution analysis
# ---------------------------------------------------------------------------

st.divider()
st.subheader("📊 Distribution Analysis")

dist_feature = st.selectbox(
    "Select feature to analyze distribution:",
    available_numeric,
    format_func=lambda x: {
        "temperature_avg": "🌡️ Average Temperature",
        "temperature_min": "🌡️ Min Temperature",
        "temperature_max": "🌡️ Max Temperature",
        "humidity": "💧 Humidity",
        "wind_speed": "💨 Wind Speed",
        "precipitation": "🌧️ Precipitation",
        "energy_kwh": "⚡ Energy Consumption",
    }.get(x, x),
)

col1, col2 = st.columns(2)

with col1:
    # Descriptive statistics
    st.markdown(f"**Descriptive Statistics for {dist_feature}:**")
    desc = df[dist_feature].describe()
    st.dataframe(desc.to_frame().T, width="stretch")

with col2:
    # Monthly average for the selected feature
    monthly = df.copy()
    monthly["month"] = monthly["date"].dt.month_name()
    monthly["month_num"] = monthly["date"].dt.month
    monthly_avg = monthly.groupby(["month", "month_num"])[dist_feature].mean().reset_index()
    monthly_avg = monthly_avg.sort_values("month_num")

    fig_monthly = bar_chart(
        monthly_avg, x="month", y=dist_feature,
        title=f"Monthly Average: {dist_feature.replace('_', ' ').title()}",
    )
    st.plotly_chart(fig_monthly, width="stretch")


# ---------------------------------------------------------------------------
# City comparison
# ---------------------------------------------------------------------------

if selected_city == "All" and df["city"].nunique() > 1:
    st.divider()
    st.subheader("🏙️ City Comparison")

    comparison = df.groupby("city").agg({
        "energy_kwh": ["mean", "std", "min", "max"],
        "temperature_avg": "mean",
    }).round(2)
    comparison.columns = ["Avg Energy (kWh)", "Std Dev", "Min kWh", "Max kWh", "Avg Temp (°C)"]
    comparison = comparison.reset_index()

    st.dataframe(comparison, width="stretch", hide_index=True)
