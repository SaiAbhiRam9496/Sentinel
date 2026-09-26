"""
ev_cleaning.py — Phase 5
EV-specific cleaning rules layered on top of the generic engine.
Drops rows that are missing core EV identity/measurement fields.
Median-imputes remaining numeric gaps.
All actions logged to CleaningLog — identical transparency contract as generic mode.
"""

import pandas as pd
from typing import Tuple, List
from sqlalchemy.orm import Session
from backend.app.db.models import CleaningLog


# Columns whose absence makes a session record uninterpretable
REQUIRED_EV_FIELDS = [
    "User ID",
    "Charging Start Time",
    "Charging End Time",
    "Energy Consumed (kWh)",
]


def _log(db: Session, dataset_id: str, action: str, details: dict) -> None:
    db.add(CleaningLog(dataset_id=dataset_id, action=action, details=details))
    db.commit()


def drop_missing_core_ev(df: pd.DataFrame, dataset_id: str, db: Session) -> pd.DataFrame:
    """Drop rows missing any required EV field.
    Per handoff rule: missing a field essential to identifying/interpreting the record → drop.
    """
    present_required = [c for c in REQUIRED_EV_FIELDS if c in df.columns]
    if not present_required:
        return df

    missing_core_mask = df[present_required].isna().any(axis=1)
    dropped_indices = df[missing_core_mask].index.tolist()
    if dropped_indices:
        _log(db, dataset_id, "drop", {
            "rows": dropped_indices,
            "reason": "Missing one or more required EV fields: " + ", ".join(present_required),
            "fields_checked": present_required,
        })
        df = df[~missing_core_mask]
    return df


def impute_numeric_ev(df: pd.DataFrame, dataset_id: str, db: Session) -> pd.DataFrame:
    """Median-impute remaining numeric gaps after required-field drops."""
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            imputed_rows = df[df[col].isna()].index.tolist()
            df[col] = df[col].fillna(median_val)
            _log(db, dataset_id, "impute", {
                "column": col,
                "rows": imputed_rows,
                "method": "median",
                "value": float(median_val),
                "mode": "ev",
            })
    return df


def clean_ev_dataset(df: pd.DataFrame, dataset_id: str, db: Session) -> Tuple[pd.DataFrame, int, int]:
    """Full EV cleaning pipeline: drop required-field gaps → median-impute numeric gaps.
    Returns (cleaned_df, rows_dropped, rows_imputed_total).
    """
    original_len = len(df)
    df = drop_missing_core_ev(df, dataset_id, db)
    rows_dropped = original_len - len(df)
    df = impute_numeric_ev(df, dataset_id, db)
    return df, rows_dropped
