"""
Machine Learning module for the Weather & Energy Optimizer.

Implements a Random Forest Regressor to predict daily energy consumption
(kWh) based on weather features. This helps households anticipate their
energy needs and optimize usage.

Model pipeline:
    1. Feature engineering (extract month, day of week from dates)
    2. Train/test split (80/20)
    3. Train RandomForestRegressor
    4. Evaluate with R², MAE, RMSE
    5. Predict energy consumption from new weather inputs
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

# These are the weather features used as inputs to the model
FEATURE_COLUMNS = [
    "temperature_avg",
    "temperature_min",
    "temperature_max",
    "humidity",
    "wind_speed",
    "precipitation",
    "month",
    "day_of_week",
]

# Target column: what we are predicting
TARGET_COLUMN = "energy_kwh"


def prepare_features(df):
    """
    Prepare the feature matrix from raw energy records.

    Extracts temporal features (month, day of week) from the date column
    and selects the weather feature columns.

    Args:
        df (pd.DataFrame): Raw energy records with 'date' column

    Returns:
        tuple: (X, y) where X is the feature DataFrame and y is the target Series.
               Returns (None, None) if data is insufficient.
    """
    if df.empty or len(df) < 10:
        return None, None

    # Make a copy to avoid modifying the original
    data = df.copy()

    # Convert date to datetime if it's a string
    data["date"] = pd.to_datetime(data["date"])

    # Extract temporal features from date
    data["month"] = data["date"].dt.month
    data["day_of_week"] = data["date"].dt.dayofweek  # 0=Monday, 6=Sunday

    # Drop rows with missing values in feature or target columns
    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    data = data.dropna(subset=required_cols)

    if len(data) < 10:
        return None, None

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    return X, y


# ---------------------------------------------------------------------------
# Model training and evaluation
# ---------------------------------------------------------------------------

def train_model(df, test_size=0.2, random_state=42):
    """
    Train a Random Forest Regressor on the provided energy records.

    The model learns the relationship between weather conditions and
    energy consumption, enabling predictions for future weather scenarios.

    Args:
        df (pd.DataFrame): Energy records with weather features and energy_kwh
        test_size (float): Fraction of data to use for testing (default 0.2)
        random_state (int): Random seed for reproducibility

    Returns:
        dict: Contains:
            - 'model': Trained RandomForestRegressor
            - 'metrics': Dict with 'r2', 'mae', 'rmse' scores
            - 'X_test': Test features
            - 'y_test': True test values
            - 'y_pred': Predicted test values
            - 'feature_importance': DataFrame of feature importances
            - 'scaler': Fitted StandardScaler (for future predictions)

        Returns None if training data is insufficient.
    """
    # Prepare features
    X, y = prepare_features(df)
    if X is None:
        return None

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale features for more stable training
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the Random Forest model
    # n_estimators=100: number of trees in the forest
    # max_depth=10: limit tree depth to prevent overfitting
    # random_state: ensures reproducible results
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1,  # Use all available CPU cores for training
    )
    model.fit(X_train_scaled, y_train)

    # Generate predictions on the test set
    y_pred = model.predict(X_test_scaled)

    # Calculate evaluation metrics
    metrics = evaluate_model(y_test, y_pred)

    # Calculate feature importance
    importance_df = get_feature_importance(model, FEATURE_COLUMNS)

    return {
        "model": model,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "feature_importance": importance_df,
        "scaler": scaler,
    }


def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance using standard regression metrics.

    Args:
        y_true (array-like): Actual energy consumption values
        y_pred (array-like): Predicted energy consumption values

    Returns:
        dict: Evaluation metrics:
            - 'r2': R-squared score (1.0 = perfect, 0.0 = baseline)
            - 'mae': Mean Absolute Error (average prediction error in kWh)
            - 'rmse': Root Mean Squared Error (penalizes large errors)
    """
    return {
        "r2": round(r2_score(y_true, y_pred), 4),
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
    }


def get_feature_importance(model, feature_names):
    """
    Extract and sort feature importances from the trained model.

    Feature importance indicates how much each weather variable
    contributes to the energy consumption prediction.

    Args:
        model: Trained RandomForestRegressor
        feature_names (list): Names of the feature columns

    Returns:
        pd.DataFrame: Sorted feature importances with columns
                       ['feature', 'importance']
    """
    importance = model.feature_importances_
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    })
    # Sort by importance descending
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_energy(model, scaler, temp_avg, temp_min, temp_max,
                   humidity, wind_speed, precipitation, month, day_of_week):
    """
    Predict energy consumption for a single set of weather conditions.

    This is used in the prediction form where users input weather values
    and receive an estimated energy consumption.

    Args:
        model: Trained RandomForestRegressor
        scaler: Fitted StandardScaler
        temp_avg (float): Average temperature (°C)
        temp_min (float): Minimum temperature (°C)
        temp_max (float): Maximum temperature (°C)
        humidity (float): Relative humidity (%)
        wind_speed (float): Wind speed (km/h)
        precipitation (float): Precipitation (mm)
        month (int): Month of year (1-12)
        day_of_week (int): Day of week (0=Mon, 6=Sun)

    Returns:
        float: Predicted energy consumption in kWh
    """
    # Create a single-row DataFrame with the input features
    input_data = pd.DataFrame([{
        "temperature_avg": temp_avg,
        "temperature_min": temp_min,
        "temperature_max": temp_max,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "precipitation": precipitation,
        "month": month,
        "day_of_week": day_of_week,
    }])

    # Scale features using the same scaler used during training
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Ensure prediction is non-negative (energy can't be negative)
    return max(0.0, round(prediction, 2))
