"""
ML Predictions Page – Train, Evaluate, and Use the Energy Model

This page allows users to:
    1. Train a Random Forest model on the energy consumption dataset
    2. View model evaluation metrics (R², MAE, RMSE)
    3. See feature importance rankings
    4. Compare predicted vs actual values
    5. Make new predictions by entering weather conditions
    6. Predict energy from the weather forecast
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.database import get_all_records, get_cities
from utils.ml_model import train_model, predict_energy, FEATURE_COLUMNS, TARGET_COLUMN
from utils.api_client import fetch_weather_forecast
from utils.charts import (
    prediction_vs_actual_chart, feature_importance_chart,
    bar_chart, gauge_chart, COLORS,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="🤖 ML Predictions",
    page_icon="🤖",
    layout="wide",
)

st.markdown("# 🤖 Machine Learning – Energy Prediction")
st.markdown("""
Train a **Random Forest Regressor** to predict daily energy consumption (kWh)
based on weather features. The model learns the relationship between temperature,
humidity, wind, precipitation, and energy usage.
""")


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.markdown("## 🧠 Model Settings")

test_size = st.sidebar.slider(
    "Test set size (%)",
    min_value=10, max_value=40, value=20, step=5,
    help="Percentage of data used for testing (not training)",
)

random_state = st.sidebar.number_input(
    "Random seed",
    min_value=0, max_value=999, value=42,
    help="Seed for reproducibility",
)

st.sidebar.divider()

# City filter for predictions
cities_df = get_cities()
city_names = cities_df["name"].tolist() if not cities_df.empty else ["St. Gallen"]
predict_city = st.sidebar.selectbox("🏙️ Prediction City", city_names, index=0)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df = get_all_records()

if df.empty or len(df) < 20:
    st.error("❌ Not enough data to train the model. Need at least 20 records. "
             "Please add data first via the 📝 Add Data page.")
    st.stop()


# ---------------------------------------------------------------------------
# Model training section
# ---------------------------------------------------------------------------

st.subheader("🏋️ Model Training")

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown(f"""
    - **Algorithm**: Random Forest Regressor (100 trees, max depth 10)
    - **Features**: {', '.join(FEATURE_COLUMNS)}
    - **Target**: {TARGET_COLUMN}
    - **Dataset size**: {len(df):,} records
    """)
with col2:
    st.metric("Training Set", f"{100 - test_size}%")
with col3:
    st.metric("Test Set", f"{test_size}%")

# Train button
train_clicked = st.button("🚀 Train Model", type="primary", width="stretch")

# Use session state to persist model results across reruns
if "model_results" not in st.session_state:
    st.session_state.model_results = None

if train_clicked:
    with st.spinner("Training Random Forest model... ⏳"):
        results = train_model(df, test_size=test_size / 100, random_state=random_state)

    if results is None:
        st.error("❌ Training failed. Check that the data has enough valid records "
                 "with all required features.")
    else:
        st.session_state.model_results = results
        st.success("✅ Model trained successfully!")


# ---------------------------------------------------------------------------
# Display results if model is trained
# ---------------------------------------------------------------------------

results = st.session_state.model_results

if results is not None:
    metrics = results["metrics"]

    st.divider()
    st.subheader("📊 Model Evaluation")

    # --- Metric cards ---
    col1, col2, col3 = st.columns(3)

    with col1:
        fig_r2 = gauge_chart(
            metrics["r2"] * 100,
            title="R² Score",
            min_val=0, max_val=100, suffix="%",
        )
        st.plotly_chart(fig_r2, width="stretch")
        st.caption("R² measures how well the model explains variance. "
                   "100% = perfect, 0% = baseline.")

    with col2:
        fig_mae = gauge_chart(
            metrics["mae"],
            title="MAE (kWh)",
            min_val=0, max_val=10, suffix=" kWh",
        )
        st.plotly_chart(fig_mae, width="stretch")
        st.caption("Mean Absolute Error: average prediction error in kWh.")

    with col3:
        fig_rmse = gauge_chart(
            metrics["rmse"],
            title="RMSE (kWh)",
            min_val=0, max_val=15, suffix=" kWh",
        )
        st.plotly_chart(fig_rmse, width="stretch")
        st.caption("Root Mean Squared Error: penalizes large errors more than MAE.")

    # --- Metrics summary table ---
    st.markdown("### 📋 Metrics Summary")
    metrics_df = pd.DataFrame([{
        "Metric": "R² Score",
        "Value": f"{metrics['r2']:.4f}",
        "Interpretation": "Excellent" if metrics['r2'] > 0.8 else
                          "Good" if metrics['r2'] > 0.6 else "Moderate",
    }, {
        "Metric": "MAE",
        "Value": f"{metrics['mae']:.4f} kWh",
        "Interpretation": f"Predictions are off by ~{metrics['mae']:.1f} kWh on average",
    }, {
        "Metric": "RMSE",
        "Value": f"{metrics['rmse']:.4f} kWh",
        "Interpretation": f"Typical error magnitude: {metrics['rmse']:.1f} kWh",
    }])
    st.dataframe(metrics_df, width="stretch", hide_index=True)

    # --- Predicted vs Actual chart ---
    st.markdown("### 🎯 Predicted vs Actual")
    fig_pred = prediction_vs_actual_chart(
        results["y_test"].values,
        results["y_pred"],
    )
    st.plotly_chart(fig_pred, width="stretch")
    st.caption("Points closer to the dashed line indicate better predictions.")

    # --- Feature importance ---
    st.markdown("### 🏆 Feature Importance")
    fig_imp = feature_importance_chart(results["feature_importance"])
    st.plotly_chart(fig_imp, width="stretch")
    st.caption("Shows which weather features have the most influence on energy consumption.")

    # --- Manual prediction ---
    st.divider()
    st.subheader("🔮 Predict Energy Consumption")
    st.markdown("Enter weather conditions to predict the expected energy consumption.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pred_temp_avg = st.number_input("🌡️ Avg Temp (°C)", value=10.0, min_value=-20.0, max_value=45.0, step=0.5)
        pred_temp_min = st.number_input("🌡️ Min Temp (°C)", value=5.0, min_value=-30.0, max_value=40.0, step=0.5)

    with col2:
        pred_temp_max = st.number_input("🌡️ Max Temp (°C)", value=15.0, min_value=-15.0, max_value=50.0, step=0.5)
        pred_humidity = st.number_input("💧 Humidity (%)", value=70.0, min_value=0.0, max_value=100.0, step=1.0)

    with col3:
        pred_wind = st.number_input("💨 Wind Speed (km/h)", value=10.0, min_value=0.0, max_value=100.0, step=0.5)
        pred_precip = st.number_input("🌧️ Precipitation (mm)", value=2.0, min_value=0.0, max_value=100.0, step=0.5)

    with col4:
        pred_month = st.selectbox("📅 Month", list(range(1, 13)),
                                  index=datetime.today().month - 1,
                                  format_func=lambda m: datetime(2024, m, 1).strftime("%B"))
        pred_dow = st.selectbox("📅 Day of Week", list(range(7)),
                                format_func=lambda d: ["Monday", "Tuesday", "Wednesday",
                                                        "Thursday", "Friday", "Saturday", "Sunday"][d])

    if st.button("⚡ Predict Energy", type="primary"):
        prediction = predict_energy(
            results["model"], results["scaler"],
            pred_temp_avg, pred_temp_min, pred_temp_max,
            pred_humidity, pred_wind, pred_precip,
            pred_month, pred_dow,
        )
        st.success(f"🔋 **Predicted Energy Consumption: {prediction:.2f} kWh**")

        # Context: compare to average
        avg_energy = df["energy_kwh"].mean()
        diff = prediction - avg_energy
        if diff > 0:
            st.info(f"This is **{abs(diff):.1f} kWh above** the dataset average ({avg_energy:.1f} kWh).")
        else:
            st.info(f"This is **{abs(diff):.1f} kWh below** the dataset average ({avg_energy:.1f} kWh).")

    # --- Forecast-based predictions ---
    st.divider()
    st.subheader(f"🌤️ Forecast Predictions – {predict_city}")
    st.markdown("Uses the 7-day weather forecast to predict energy consumption for each upcoming day.")

    if st.button("📡 Fetch Forecast & Predict"):
        city_row = cities_df[cities_df["name"] == predict_city]
        if not city_row.empty:
            lat = city_row.iloc[0]["latitude"]
            lon = city_row.iloc[0]["longitude"]

            with st.spinner("Fetching forecast and predicting..."):
                forecast_df = fetch_weather_forecast(lat, lon, days=7)

            if not forecast_df.empty:
                # Predict for each forecast day
                predictions = []
                for _, row in forecast_df.iterrows():
                    pred = predict_energy(
                        results["model"], results["scaler"],
                        row["temperature_avg"], row["temperature_min"],
                        row["temperature_max"], row["humidity"],
                        row["wind_speed"], row["precipitation"],
                        row["date"].month, row["date"].weekday(),
                    )
                    predictions.append(pred)

                forecast_df["predicted_energy_kwh"] = predictions
                forecast_df["date_str"] = forecast_df["date"].dt.strftime("%a, %b %d")

                # Display results
                display_cols = ["date_str", "temperature_avg", "humidity",
                                "precipitation", "predicted_energy_kwh"]
                display_df = forecast_df[display_cols].copy()
                display_df.columns = ["Date", "Temp (°C)", "Humidity (%)",
                                      "Rain (mm)", "Predicted kWh"]

                st.dataframe(display_df, width="stretch", hide_index=True)

                # Bar chart of predictions
                fig_forecast = bar_chart(
                    forecast_df, x="date_str", y="predicted_energy_kwh",
                    title="Predicted Daily Energy Consumption (Next 7 Days)",
                )
                st.plotly_chart(fig_forecast, width="stretch")

                total = sum(predictions)
                st.info(f"📊 **Total predicted energy for next 7 days: {total:.1f} kWh** "
                        f"(avg: {total/7:.1f} kWh/day)")
            else:
                st.warning("Could not fetch forecast data. Try again later or check your connection.")

else:
    st.info("👆 Click **Train Model** above to get started. The model needs to be trained before "
            "you can view evaluation metrics or make predictions.")
