"""
Weather & Energy Consumption Optimizer – Main App Page

This is the home page of the Streamlit application. It displays:
    - A welcoming overview of the project
    - Key Performance Indicator (KPI) metric cards
    - A quick summary chart
    - Navigation hints to other pages

The app helps Swiss households understand how weather patterns affect
their energy consumption and predict future energy needs using ML.
"""

import streamlit as st
import pandas as pd
import os
import sys

# Add project root to Python path for imports
sys.path.insert(0, os.path.dirname(__file__))

from utils.database import init_db, seed_cities, get_all_records, get_cities, get_record_count
from utils.charts import line_chart, COLORS
from data.init_db import generate_synthetic_records, seed_records

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="⚡ Weather & Energy Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for polished appearance
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* KPI card styling */
    .kpi-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #2d3548 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(76, 175, 80, 0.2);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4CAF50;
        margin: 5px 0;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #b0b0b0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #0e1117 100%);
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 25px;
        border: 1px solid rgba(76, 175, 80, 0.15);
    }

    /* Sidebar styling */
    .css-1d391kg { padding-top: 1rem; }

    /* Feature card */
    .feature-card {
        background: #1a1f2e;
        border-radius: 10px;
        padding: 15px;
        margin: 8px 0;
        border-left: 3px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Database initialization (runs once on first load)
# ---------------------------------------------------------------------------

@st.cache_resource
def initialize_database():
    """
    Initialize the database and seed it with sample data if empty.
    Uses st.cache_resource so this only runs once per app session.
    """
    init_db()
    seed_cities()
    count = get_record_count()
    if count == 0:
        # Database is empty – seed with synthetic data
        records = generate_synthetic_records(days=365)
        seed_records(records)
    return True


# Initialize DB
initialize_database()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚡ Navigation")
    st.markdown("""
    **Pages:**
    - 🏠 **Home** – Overview & KPIs
    - 📊 **Dashboard** – Weather & energy charts
    - 📈 **Data Explorer** – Filter & explore data
    - 🤖 **ML Predictions** – Train model & predict
    - 📝 **Add Data** – Add new records
    """)

    st.divider()

    st.markdown("### 🏙️ About")
    st.markdown("""
    This app analyzes how weather affects energy
    consumption in Swiss cities and predicts future
    energy needs using Machine Learning.
    """)

    st.divider()
    st.caption("University of St. Gallen – GMI Group Project")


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

# Hero header
st.markdown("""
<div class="main-header">
    <h1>⚡ Weather & Energy Consumption Optimizer</h1>
    <p style="font-size: 1.1rem; color: #b0b0b0;">
        Understand how weather impacts your energy usage and predict future consumption
        using Machine Learning. Built for Swiss households.
    </p>
</div>
""", unsafe_allow_html=True)


# --- KPI Cards ---

df = get_all_records()

if not df.empty:
    col1, col2, col3, col4 = st.columns(4)

    avg_energy = df["energy_kwh"].mean()
    total_records = len(df)
    avg_temp = df["temperature_avg"].mean()
    cities_count = df["city"].nunique()

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Avg Daily Energy</div>
            <div class="kpi-value">{avg_energy:.1f} kWh</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Records</div>
            <div class="kpi-value">{total_records:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Avg Temperature</div>
            <div class="kpi-value">{avg_temp:.1f}°C</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Cities Tracked</div>
            <div class="kpi-value">{cities_count}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Quick overview chart ---
    st.subheader("📈 Energy Consumption Trend (Last 30 Days)")

    # Show most recent 30 days for all cities
    df["date"] = pd.to_datetime(df["date"])
    recent = df.sort_values("date").tail(30 * cities_count)

    # Aggregate by date (average across cities)
    daily_avg = recent.groupby("date").agg({
        "energy_kwh": "mean",
        "temperature_avg": "mean",
    }).reset_index()

    fig = line_chart(
        daily_avg, x="date", y="energy_kwh",
        title="Average Daily Energy Consumption Across All Cities",
        y_label="Energy (kWh)",
    )
    st.plotly_chart(fig, width="stretch")

else:
    st.info("🗄️ No data in database yet. Go to the **📝 Add Data** page to add records, "
            "or the database will be seeded automatically on next reload.")

# --- Feature overview ---
st.subheader("🔍 What This App Does")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h4>📊 Weather & Energy Dashboard</h4>
        <p>Visualize temperature, humidity, wind speed, and precipitation alongside
        energy consumption trends. Compare cities and time periods.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-card">
        <h4>📈 Data Explorer</h4>
        <p>Filter data by city, date range, and weather conditions.
        View correlation matrices and summary statistics.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h4>🤖 ML Predictions</h4>
        <p>Train a Random Forest model on historical data. Evaluate performance
        with R², MAE, and RMSE. Predict energy consumption from weather forecasts.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-card">
        <h4>📝 Add Data</h4>
        <p>Manually add energy consumption records with weather data.
        New entries are immediately reflected in all charts and analyses.</p>
    </div>
    """, unsafe_allow_html=True)

# --- Data sources info ---
st.divider()
st.subheader("📡 Data Sources")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **🌐 API: Open-Meteo**
    - Free weather API (no API key needed)
    - Historical weather data + 7-day forecasts
    - Endpoint: `api.open-meteo.com`
    - Automatic fallback to cached data when offline
    """)

with col2:
    st.markdown("""
    **🗃️ Database: SQLite**
    - Lightweight, file-based database
    - Tables: `cities`, `energy_records`
    - Pre-seeded with 365 days × 5 Swiss cities
    - File: `data/energy.db`
    """)
