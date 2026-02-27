"""
Database initialization script for the Weather & Energy Optimizer.

Run this script to create and seed the SQLite database with:
    - 5 Swiss cities (St. Gallen, Zürich, Bern, Basel, Luzern)
    - ~365 days × 5 cities = ~1825 synthetic energy consumption records

The synthetic data uses realistic seasonal patterns:
    - Energy consumption is higher in cold months (heating)
    - Temperature follows Swiss seasonal averages
    - Random noise adds natural variation

Usage:
    python data/init_db.py
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path so we can import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.database import init_db, seed_cities, _get_connection


def generate_synthetic_records(days=365):
    """
    Generate realistic synthetic energy consumption records.

    The energy model simulates a typical Swiss household where:
        - Base consumption is ~15 kWh/day
        - Cold weather increases consumption (heating)
        - Higher humidity slightly increases consumption
        - Wind has marginal effect
        - Weekends have slightly higher consumption (people at home)

    Args:
        days (int): Number of days of data to generate

    Returns:
        list[dict]: List of record dictionaries ready for DB insertion
    """
    np.random.seed(42)  # Reproducibility

    # Swiss cities with coordinates
    cities = [
        {"name": "St. Gallen", "lat": 47.42, "lon": 9.37},
        {"name": "Zürich", "lat": 47.38, "lon": 8.54},
        {"name": "Bern", "lat": 46.95, "lon": 7.45},
        {"name": "Basel", "lat": 47.56, "lon": 7.59},
        {"name": "Luzern", "lat": 47.05, "lon": 8.31},
    ]

    records = []
    base_date = datetime.today() - timedelta(days=days)

    for city in cities:
        # Each city has slightly different characteristics
        city_offset = np.random.uniform(-1, 1)  # Random city temperature offset

        for day_num in range(days):
            date = base_date + timedelta(days=day_num)
            day_of_year = date.timetuple().tm_yday
            day_of_week = date.weekday()  # 0=Monday, 6=Sunday

            # --- Seasonal temperature pattern (Swiss climate) ---
            # Peak summer ~22°C in July, winter ~-1°C in January
            seasonal_temp = 10 + 12 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
            temp_avg = seasonal_temp + city_offset + np.random.normal(0, 2.5)
            temp_min = temp_avg - 3 - abs(np.random.normal(0, 1.5))
            temp_max = temp_avg + 4 + abs(np.random.normal(0, 1.5))

            # --- Weather features ---
            humidity = np.clip(70 + 10 * np.sin((day_of_year - 30) * 2 * np.pi / 365)
                               + np.random.normal(0, 10), 30, 100)
            wind_speed = np.clip(np.random.exponential(8) + 2, 0, 45)
            precipitation = max(0, np.random.exponential(2.5) - 0.5)
            if np.random.random() < 0.3:  # 30% chance of dry day
                precipitation = 0.0

            # --- Energy consumption model ---
            # Base consumption: 15 kWh
            energy = 15.0

            # Temperature effect: colder → more energy (heating)
            # Below 15°C: each degree costs ~0.8 kWh more
            if temp_avg < 15:
                energy += (15 - temp_avg) * 0.8
            # Above 25°C: slight increase (cooling/AC)
            if temp_avg > 25:
                energy += (temp_avg - 25) * 0.4

            # Humidity effect: higher humidity → slightly more energy
            energy += max(0, (humidity - 60)) * 0.03

            # Wind effect: strong wind → slightly more heating needed
            energy += wind_speed * 0.05

            # Weekend effect: people at home → more consumption
            if day_of_week >= 5:
                energy += 2.5

            # Random noise
            energy += np.random.normal(0, 1.5)
            energy = max(5.0, round(energy, 2))  # Minimum 5 kWh

            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "city": city["name"],
                "temperature_avg": round(temp_avg, 1),
                "temperature_min": round(temp_min, 1),
                "temperature_max": round(temp_max, 1),
                "humidity": round(humidity, 1),
                "wind_speed": round(wind_speed, 1),
                "precipitation": round(precipitation, 1),
                "energy_kwh": energy,
            })

    return records


def seed_records(records):
    """
    Insert synthetic records into the database.

    Uses INSERT OR IGNORE to skip duplicates if the script is run again.

    Args:
        records (list[dict]): Records to insert
    """
    conn = _get_connection()
    cursor = conn.cursor()

    inserted = 0
    for r in records:
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO energy_records
                    (date, city, temperature_avg, temperature_min, temperature_max,
                     humidity, wind_speed, precipitation, energy_kwh)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                r["date"], r["city"], r["temperature_avg"], r["temperature_min"],
                r["temperature_max"], r["humidity"], r["wind_speed"],
                r["precipitation"], r["energy_kwh"],
            ))
            if cursor.rowcount > 0:
                inserted += 1
        except Exception as e:
            print(f"Error inserting record: {e}")

    conn.commit()
    conn.close()
    return inserted


def main():
    """Main entry point: create tables, seed cities, generate and insert records."""
    print("🔧 Initializing database...")
    init_db()

    print("🏙️  Seeding cities...")
    seed_cities()

    print("📊 Generating synthetic energy records...")
    records = generate_synthetic_records(days=365)

    print(f"💾 Inserting {len(records)} records into database...")
    inserted = seed_records(records)
    print(f"✅ Done! Inserted {inserted} new records.")

    # Print summary
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM energy_records")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM cities")
    city_count = cursor.fetchone()[0]
    conn.close()

    print(f"\n📈 Database summary:")
    print(f"   Cities: {city_count}")
    print(f"   Energy records: {total}")
    print(f"   Database location: {os.path.abspath(os.path.join(os.path.dirname(__file__), 'energy.db'))}")


if __name__ == "__main__":
    main()
