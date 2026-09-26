import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

# Canonical EV Schema columns per Section 3.1 of handoff spec
CANONICAL_EV_COLUMNS = [
    "User ID",
    "Vehicle Model",
    "Battery Capacity (kWh)",
    "Charging Station ID",
    "Charging Station Location",
    "Charging Start Time",
    "Charging End Time",
    "Energy Consumed (kWh)",
    "Charging Duration (hours)",
    "Charging Rate (kW)",
    "Charging Cost (USD)",
    "Time of Day",
    "Day of Week",
    "State of Charge (Start %)",
    "State of Charge (End %)",
    "Distance Driven (since last charge) (km)",
    "Temperature (°C)",
    "Vehicle Age (years)",
    "Charger Type",
    "User Type"
]

# Required signature columns that strongly differentiate EV charging telemetry
EV_SIGNATURE_COLUMNS = [
    "Battery Capacity (kWh)",
    "Energy Consumed (kWh)",
    "State of Charge (Start %)",
    "State of Charge (End %)",
    "Charging Duration (hours)"
]

def normalize_col(name: str) -> str:
    """Normalize column name for robust comparison (lowercase, strip symbols/spaces)."""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())

def infer_column_type(series: pd.Series) -> str:
    """Infers high-level data type of a column (integer, float, boolean, datetime, categorical, text)."""
    # Drop NAs for type inspection
    non_null = series.dropna()
    if len(non_null) == 0:
        return "empty"

    # Check for boolean
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if set(non_null.astype(str).str.lower().unique()).issubset({"true", "false", "0", "1", "yes", "no"}):
        if len(non_null.unique()) <= 2:
            return "boolean"

    # Check numeric types
    if pd.api.types.is_numeric_dtype(series):
        # Check if all integers
        try:
            if (non_null % 1 == 0).all():
                return "integer"
            return "float"
        except Exception:
            return "float"

    # Try datetime inference
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    
    # Try parsing sample strings as datetime if column looks like a date/time
    sample = non_null.head(20).astype(str)
    date_patterns = [r'\d{4}-\d{2}-\d{2}', r'\d{2}/\d{2}/\d{4}', r'\d{4}/\d{2}/\d{2}', r'\d{2}:\d{2}:\d{2}']
    if any(sample.str.contains(pat).any() for pat in date_patterns):
        try:
            pd.to_datetime(sample, errors='raise', format='mixed')
            return "datetime"
        except Exception:
            pass

    # Check categorical vs free text
    unique_count = len(non_null.unique())
    total_count = len(non_null)
    if unique_count <= 50 and (unique_count / total_count < 0.25 or unique_count <= 10):
        return "categorical"

    return "text"

def detect_schema_and_types(df: pd.DataFrame) -> Dict[str, Any]:
    """Inspects all columns and infers types and basic summary metrics."""
    column_types = {}
    for col in df.columns:
        col_type = infer_column_type(df[col])
        column_types[str(col)] = {
            "type": col_type,
            "non_null_count": int(df[col].count()),
            "null_count": int(df[col].isna().sum()),
            "unique_count": int(df[col].nunique())
        }
    return column_types

def detect_mode(df: pd.DataFrame) -> Tuple[str, str, Dict[str, Any]]:
    """
    Evaluates whether the dataframe qualifies for EV Mode or Generic Mode.
    Returns (mode, reason, match_details).
    """
    norm_uploaded = {normalize_col(c): c for c in df.columns}
    norm_ev = {normalize_col(c): c for c in CANONICAL_EV_COLUMNS}
    norm_signature = {normalize_col(c): c for c in EV_SIGNATURE_COLUMNS}

    matched_canonical = [norm_ev[k] for k in norm_ev if k in norm_uploaded]
    matched_signatures = [norm_signature[k] for k in norm_signature if k in norm_uploaded]
    
    total_canonical = len(CANONICAL_EV_COLUMNS)
    match_pct = (len(matched_canonical) / total_canonical) * 100.0

    match_details = {
        "matched_canonical_columns": matched_canonical,
        "matched_canonical_count": len(matched_canonical),
        "total_canonical_count": total_canonical,
        "canonical_match_percentage": round(match_pct, 1),
        "matched_signature_columns": matched_signatures,
        "signature_match_count": len(matched_signatures),
        "total_signature_count": len(EV_SIGNATURE_COLUMNS)
    }

    # Qualifying threshold: >= 75% canonical columns match AND at least 3 core signature columns match
    if match_pct >= 75.0 and len(matched_signatures) >= 3:
        mode = "ev"
        reason = (
            f"EV Charging Telemetry detected: matched {len(matched_canonical)}/{total_canonical} "
            f"canonical EV columns ({round(match_pct, 1)}% match) including core battery and charging telemetry fields."
        )
    else:
        mode = "generic"
        if len(matched_canonical) > 0:
            reason = (
                f"Generic dataset detected: matched only {len(matched_canonical)}/{total_canonical} "
                f"canonical EV columns ({round(match_pct, 1)}% match). Running universal data quality profiling."
            )
        else:
            reason = "Generic structured dataset detected (0% EV canonical match). Running universal data quality profiling."

    return mode, reason, match_details
