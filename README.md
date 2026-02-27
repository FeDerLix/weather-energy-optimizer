# ⚡ Weather & Energy Consumption Optimizer

A Streamlit web application that analyzes how weather patterns affect household energy consumption in Swiss cities and predicts future energy needs using Machine Learning.

> **University of St. Gallen** – *Grundlagen und Methoden der Informatik* – Group Project

---

## 📋 Problem Statement

Swiss households face unpredictable energy costs driven largely by weather conditions. This application solves the problem by:

1. **Visualizing** the relationship between weather (temperature, humidity, wind, precipitation) and daily energy consumption
2. **Predicting** future energy consumption based on weather forecasts using a trained ML model
3. **Empowering** users to plan energy usage, compare cities, and understand consumption drivers

---

## 🏗️ Architecture Overview

```
weather-energy-optimizer/
│
├── app.py                           # 🏠 Home page: KPI cards, overview, trend chart
│
├── pages/
│   ├── 1_📊_Dashboard.py            # Weather + energy charts, dual-axis, forecast
│   ├── 2_📈_Data_Explorer.py        # Filter, explore, correlations, distributions
│   ├── 3_🤖_ML_Predictions.py       # Train model, evaluate, predict
│   └── 4_📝_Add_Data.py             # Add records form, auto-fill from API
│
├── utils/
│   ├── __init__.py
│   ├── api_client.py                # Open-Meteo API client + offline fallback
│   ├── database.py                  # SQLite database CRUD operations
│   ├── ml_model.py                  # RandomForest training, evaluation, prediction
│   └── charts.py                    # Plotly chart builders (consistent theme)
│
├── data/
│   ├── init_db.py                   # Database initialization & seed script
│   └── energy.db                    # SQLite database (auto-created)
│
├── .streamlit/
│   └── config.toml                  # Theme & server configuration
│
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── CONTRIBUTIONS.md                 # Team contribution matrix
```

### Data Flow

```
Open-Meteo API ──→ api_client.py ──→ Dashboard / Add Data
                                  ↗
SQLite DB ←──→ database.py ───────→ All pages
                                  ↘
energy_records ──→ ml_model.py ──→ ML Predictions page
```

---

## 🚀 Setup & Run Instructions

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Installation

```bash
# 1. Clone or download the project
cd weather-energy-optimizer

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialize the database with sample data
python data/init_db.py

# 5. Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

> **Note:** The database is automatically initialized with 1,825 sample records (365 days × 5 cities) if it's empty when the app starts. Step 4 is optional but recommended for faster first load.

### Offline Mode

The app works **fully offline** after initial setup. If the Open-Meteo API is unavailable:
- The app falls back to cached weather data
- If no cache exists, synthetic data is generated automatically
- All database-driven features work without any internet connection

---

## 📡 Data Sources

### API: Open-Meteo

| Property | Details |
|----------|---------|
| Provider | [Open-Meteo](https://open-meteo.com/) |
| Auth | **No API key required** (free & open) |
| Endpoints | `archive-api.open-meteo.com/v1/archive` (historical) |
| | `api.open-meteo.com/v1/forecast` (7-day forecast) |
| Variables | temperature (mean/min/max), humidity, wind speed, precipitation |
| Timezone | Europe/Zurich |

### Database: SQLite

**Table: `cities`**

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| name | TEXT | City name (unique) |
| latitude | REAL | GPS latitude |
| longitude | REAL | GPS longitude |
| country | TEXT | Country (default: Switzerland) |

**Table: `energy_records`**

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| date | TEXT | Date (YYYY-MM-DD) |
| city | TEXT | City name |
| temperature_avg | REAL | Average daily temperature (°C) |
| temperature_min | REAL | Minimum daily temperature (°C) |
| temperature_max | REAL | Maximum daily temperature (°C) |
| humidity | REAL | Relative humidity (%) |
| wind_speed | REAL | Max wind speed (km/h) |
| precipitation | REAL | Total precipitation (mm) |
| energy_kwh | REAL | Energy consumption (kWh) |
| created_at | TEXT | Record creation timestamp |

**Unique constraint:** (date, city) — one record per city per day.

---

## 🤖 Machine Learning

### Model: Random Forest Regressor

| Property | Value |
|----------|-------|
| Algorithm | `sklearn.ensemble.RandomForestRegressor` |
| Trees | 100 estimators |
| Max depth | 10 (prevents overfitting) |
| Features | temperature (avg/min/max), humidity, wind_speed, precipitation, month, day_of_week |
| Target | energy_kwh |
| Split | 80% train / 20% test |
| Scaling | StandardScaler applied to features |

### How It Works

1. **Feature engineering**: Extracts `month` and `day_of_week` from dates to capture seasonal and weekly patterns
2. **Training**: Fits a Random Forest on weather → energy relationships
3. **Evaluation**: Reports R², MAE, RMSE on the held-out test set
4. **Prediction**: Users input weather conditions or use the live forecast to get energy estimates

### Why Random Forest?

- Handles non-linear relationships (e.g., U-shaped energy curve: heating in winter, cooling in summer)
- Provides feature importance rankings
- Robust to outliers and doesn't require feature normalization (though we apply scaling for consistency)
- Good performance with moderate data sizes

### Expected Performance

With the synthetic dataset, typical metrics are:
- **R²**: ~0.85–0.95 (model explains most variance)
- **MAE**: ~1.0–2.0 kWh (average prediction error)
- **RMSE**: ~1.5–2.5 kWh

---

## 📌 Requirement Mapping

| # | Requirement | Where to Find |
|---|------------|---------------|
| 1 | Clearly stated problem | This README (Problem Statement section) |
| 2 | Data loaded via API | `utils/api_client.py` → Open-Meteo API |
| 3 | Data stored in DB/DBMS | `utils/database.py` → SQLite, `data/energy.db` |
| 4 | Data visualization | `pages/1_📊_Dashboard.py`, `utils/charts.py` |
| 5 | User interaction | `pages/4_📝_Add_Data.py` (forms), sidebar filters on all pages |
| 6 | ML component (train + eval + use) | `pages/3_🤖_ML_Predictions.py`, `utils/ml_model.py` |
| 7 | Well documented code | Comments in all `.py` files, docstrings on all functions |
| 8 | Contributions tracking | `CONTRIBUTIONS.md` |

---

## 🎬 4-Minute Demo Video – Talk Track Outline

### Slide 1: Introduction (30s)
- Project name & team members
- Problem: "How does weather affect energy consumption?"
- Goal: predict and optimize

### Slide 2: Live Demo – Dashboard (60s)
- Show KPI cards on home page
- Navigate to Dashboard
- Switch cities in sidebar, change date range
- Point out dual-axis chart and scatter plot

### Slide 3: Live Demo – Data (45s)
- Show Data Explorer with filters
- Point out correlation heatmap
- Navigate to Add Data, add a new record
- Show it appears in the table

### Slide 4: Live Demo – ML (60s)
- Click "Train Model"
- Show R², MAE, RMSE metrics
- Show predicted vs actual scatter
- Show feature importance
- Enter manual weather values → get prediction
- Fetch forecast → show predictions for next 7 days

### Slide 5: Architecture & Tech (30s)
- Show file structure
- Mention: Streamlit, SQLite, Open-Meteo API, scikit-learn, Plotly
- Highlight offline fallback mechanism

### Slide 6: Conclusion (15s)
- Summary of what was built
- What we learned
- Thank the audience

---

## 👥 Team

See [CONTRIBUTIONS.md](CONTRIBUTIONS.md) for the contribution matrix and work log.

---

## 📄 License

This project was created for educational purposes as part of the University of St. Gallen GMI course.
