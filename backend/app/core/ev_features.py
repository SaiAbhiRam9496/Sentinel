"""
ev_features.py — Phase 5
Feature engineering for the EV ML anomaly pipeline.
Derived features are added to the dataframe and also returned as a separate
feature matrix ready for model training/inference.
Per handoff Section 5 candidate list.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List


FEATURE_COLUMNS: List[str] = [
    "soc_change",
    "energy_consumed_kwh",
    "charging_duration_hours",
    "observed_charging_rate_kw",
    "energy_per_battery_pct",
    "hour_of_day",
    "charger_type_encoded",
    "vehicle_age_years",
    "temperature_c",
    "distance_km",
]


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add derived feature columns to df and return (enriched_df, feature_df).

    feature_df contains only the numeric feature columns ready for sklearn.
    Any row where a required source column is missing is filled with the
    column median to avoid NaNs in the feature matrix.
    """
    feat = df.copy()

    # --- SOC change (end - start) -------------------------------------------
    if "State of Charge (Start %)" in feat.columns and "State of Charge (End %)" in feat.columns:
        feat["soc_change"] = (
            pd.to_numeric(feat["State of Charge (End %)"], errors="coerce")
            - pd.to_numeric(feat["State of Charge (Start %)"], errors="coerce")
        )
    else:
        feat["soc_change"] = np.nan

    # --- Energy consumed (raw passthrough) ----------------------------------
    if "Energy Consumed (kWh)" in feat.columns:
        feat["energy_consumed_kwh"] = pd.to_numeric(feat["Energy Consumed (kWh)"], errors="coerce")
    else:
        feat["energy_consumed_kwh"] = np.nan

    # --- Charging duration (raw passthrough) ---------------------------------
    if "Charging Duration (hours)" in feat.columns:
        feat["charging_duration_hours"] = pd.to_numeric(feat["Charging Duration (hours)"], errors="coerce")
    else:
        feat["charging_duration_hours"] = np.nan

    # --- Observed effective charging rate (energy / duration) ---------------
    # Use the raw charging rate column if duration is 0 / missing
    if "Charging Rate (kW)" in feat.columns:
        feat["observed_charging_rate_kw"] = pd.to_numeric(feat["Charging Rate (kW)"], errors="coerce")
    else:
        safe_dur = feat["charging_duration_hours"].replace(0, np.nan)
        feat["observed_charging_rate_kw"] = feat["energy_consumed_kwh"] / safe_dur

    # --- Energy consumed relative to battery capacity -----------------------
    if "Battery Capacity (kWh)" in feat.columns:
        battery = pd.to_numeric(feat["Battery Capacity (kWh)"], errors="coerce").replace(0, np.nan)
        feat["energy_per_battery_pct"] = (feat["energy_consumed_kwh"] / battery) * 100
    else:
        feat["energy_per_battery_pct"] = np.nan

    # --- Hour of day (from Charging Start Time) -----------------------------
    if "Charging Start Time" in feat.columns:
        try:
            parsed = pd.to_datetime(feat["Charging Start Time"], errors="coerce")
            feat["hour_of_day"] = parsed.dt.hour.astype(float)
        except Exception:
            feat["hour_of_day"] = np.nan
    else:
        feat["hour_of_day"] = np.nan

    # --- Charger type encoded (label encode alphabetically) -----------------
    if "Charger Type" in feat.columns:
        categories = sorted(feat["Charger Type"].dropna().unique().tolist())
        mapping = {v: i for i, v in enumerate(categories)}
        feat["charger_type_encoded"] = feat["Charger Type"].map(mapping).astype(float)
    else:
        feat["charger_type_encoded"] = np.nan

    # --- Vehicle age --------------------------------------------------------
    if "Vehicle Age (years)" in feat.columns:
        feat["vehicle_age_years"] = pd.to_numeric(feat["Vehicle Age (years)"], errors="coerce")
    else:
        feat["vehicle_age_years"] = np.nan

    # --- Temperature --------------------------------------------------------
    if "Temperature (°C)" in feat.columns:
        feat["temperature_c"] = pd.to_numeric(feat["Temperature (°C)"], errors="coerce")
    else:
        feat["temperature_c"] = np.nan

    # --- Distance driven ----------------------------------------------------
    col_name = "Distance Driven (since last charge) (km)"
    if col_name in feat.columns:
        feat["distance_km"] = pd.to_numeric(feat[col_name], errors="coerce")
    else:
        feat["distance_km"] = np.nan

    # --- Build feature matrix -----------------------------------------------
    feature_df = feat[FEATURE_COLUMNS].copy()

    # Fill remaining NaNs column-wise with the column median (keeps matrix clean for sklearn)
    for col in FEATURE_COLUMNS:
        if feature_df[col].isna().any():
            median = feature_df[col].median()
            feature_df[col].fillna(median if pd.notna(median) else 0.0, inplace=True)

    return feat, feature_df
