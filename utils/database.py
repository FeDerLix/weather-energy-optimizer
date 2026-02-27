"""
Database module for the Weather & Energy Optimizer.

Handles all SQLite database operations including:
- Database initialization and table creation
- CRUD operations for energy consumption records
- City management
- Filtered queries for data exploration

Uses sqlite3 for lightweight, file-based persistence.
"""

import sqlite3
import os
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to the SQLite database file (stored in the data/ directory)
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "energy.db")


def _get_connection():
    """
    Create and return a new database connection.
    Ensures the data directory exists before connecting.

    Returns:
        sqlite3.Connection: Active database connection
    """
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable column-name access on rows
    return conn


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------

def init_db():
    """
    Create the database tables if they do not exist.

    Tables:
        cities  – Reference table of Swiss cities with coordinates
        energy_records – Daily energy consumption records with weather data
    """
    conn = _get_connection()
    cursor = conn.cursor()

    # Cities table: stores location info for API calls
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            country TEXT NOT NULL DEFAULT 'Switzerland'
        )
    """)

    # Energy records table: one row per city per day
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS energy_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            city TEXT NOT NULL,
            temperature_avg REAL,
            temperature_min REAL,
            temperature_max REAL,
            humidity REAL,
            wind_speed REAL,
            precipitation REAL,
            energy_kwh REAL NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(date, city)
        )
    """)

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# City operations
# ---------------------------------------------------------------------------

def seed_cities():
    """
    Insert the default set of Swiss cities if they are not already present.
    These cities are chosen because St. Gallen is the university location.
    """
    cities = [
        ("St. Gallen", 47.42, 9.37, "Switzerland"),
        ("Zürich", 47.38, 8.54, "Switzerland"),
        ("Bern", 46.95, 7.45, "Switzerland"),
        ("Basel", 47.56, 7.59, "Switzerland"),
        ("Luzern", 47.05, 8.31, "Switzerland"),
    ]
    conn = _get_connection()
    cursor = conn.cursor()
    for name, lat, lon, country in cities:
        cursor.execute(
            "INSERT OR IGNORE INTO cities (name, latitude, longitude, country) VALUES (?, ?, ?, ?)",
            (name, lat, lon, country),
        )
    conn.commit()
    conn.close()


def get_cities():
    """
    Retrieve all cities from the database.

    Returns:
        pd.DataFrame: DataFrame with columns [id, name, latitude, longitude, country]
    """
    conn = _get_connection()
    df = pd.read_sql_query("SELECT * FROM cities ORDER BY name", conn)
    conn.close()
    return df


# ---------------------------------------------------------------------------
# Energy record operations
# ---------------------------------------------------------------------------

def add_record(date, city, temp_avg, temp_min, temp_max, humidity, wind_speed,
               precipitation, energy_kwh):
    """
    Insert a single energy consumption record into the database.

    Args:
        date (str): Date string in YYYY-MM-DD format
        city (str): City name (must exist in cities table)
        temp_avg (float): Average temperature in °C
        temp_min (float): Minimum temperature in °C
        temp_max (float): Maximum temperature in °C
        humidity (float): Relative humidity in %
        wind_speed (float): Wind speed in km/h
        precipitation (float): Precipitation in mm
        energy_kwh (float): Energy consumption in kWh

    Returns:
        bool: True if insert succeeded, False if duplicate
    """
    conn = _get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO energy_records
                (date, city, temperature_avg, temperature_min, temperature_max,
                 humidity, wind_speed, precipitation, energy_kwh)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (date, city, temp_avg, temp_min, temp_max, humidity,
              wind_speed, precipitation, energy_kwh))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Duplicate date+city combination
        return False
    finally:
        conn.close()


def get_all_records():
    """
    Retrieve all energy records ordered by date descending.

    Returns:
        pd.DataFrame: All energy records
    """
    conn = _get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM energy_records ORDER BY date DESC", conn
    )
    conn.close()
    return df


def get_records_filtered(city=None, start_date=None, end_date=None):
    """
    Retrieve energy records with optional filters.

    Args:
        city (str, optional): Filter by city name
        start_date (str, optional): Filter records on or after this date (YYYY-MM-DD)
        end_date (str, optional): Filter records on or before this date (YYYY-MM-DD)

    Returns:
        pd.DataFrame: Filtered energy records
    """
    conn = _get_connection()
    query = "SELECT * FROM energy_records WHERE 1=1"
    params = []

    if city and city != "All":
        query += " AND city = ?"
        params.append(city)
    if start_date:
        query += " AND date >= ?"
        params.append(str(start_date))
    if end_date:
        query += " AND date <= ?"
        params.append(str(end_date))

    query += " ORDER BY date DESC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_record_count():
    """Return the total number of energy records in the database."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM energy_records")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_date_range():
    """
    Get the earliest and latest dates in the energy records.

    Returns:
        tuple: (min_date, max_date) as strings, or (None, None) if empty
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MIN(date), MAX(date) FROM energy_records")
    row = cursor.fetchone()
    conn.close()
    return (row[0], row[1]) if row[0] else (None, None)


def delete_record(record_id):
    """
    Delete an energy record by its ID.

    Args:
        record_id (int): The record ID to delete

    Returns:
        bool: True if a record was deleted
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM energy_records WHERE id = ?", (record_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted
