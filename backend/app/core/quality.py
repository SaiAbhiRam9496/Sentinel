import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from backend.app.db.models import Dataset, CleaningLog
from datetime import datetime, timezone


def compute_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Return per‑column profiling information.
    Includes: dtype, missing count, missing pct, unique count, unique pct, and
    basic numeric stats when applicable.
    """
    profile = {}
    total_rows = len(df)
    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        missing = int(series.isna().sum())
        missing_pct = round(missing / total_rows * 100, 2) if total_rows else 0
        unique = int(series.nunique(dropna=True))
        unique_pct = round(unique / total_rows * 100, 2) if total_rows else 0
        col_info: Dict[str, Any] = {
            "dtype": dtype,
            "missing_count": missing,
            "missing_pct": missing_pct,
            "unique_count": unique,
            "unique_pct": unique_pct,
        }
        if pd.api.types.is_numeric_dtype(series):
            col_info.update({
                "mean": round(series.mean(skipna=True), 4),
                "median": round(series.median(skipna=True), 4),
                "std": round(series.std(skipna=True), 4),
                "min": round(series.min(skipna=True), 4),
                "max": round(series.max(skipna=True), 4),
            })
        profile[col] = col_info
    return {
        "row_count": total_rows,
        "column_count": len(df.columns),
        "profile": profile,
    }


def detect_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    dup_mask = df.duplicated(keep=False)
    dup_indices = df[dup_mask].index.tolist()
    return {
        "duplicate_count": int(dup_mask.sum()),
        "duplicate_indices": dup_indices,
    }


def handle_missing_values(df: pd.DataFrame, dataset_id: str, db: Session) -> pd.DataFrame:
    """Impute numeric columns with median, drop rows where all values are missing.
    Each action is logged to the ``cleaning_logs`` table.
    """
    # Drop rows that are completely empty (all NaN)
    completely_empty = df.isna().all(axis=1)
    if completely_empty.any():
        dropped_rows = df[completely_empty].index.tolist()
        df = df[~completely_empty]
        log = CleaningLog(
            dataset_id=dataset_id,
            action="drop",
            details={"rows": dropped_rows, "reason": "all values missing"},
        )
        db.add(log)
        db.commit()

    # Impute numeric columns with median
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            imputed_rows = df[df[col].isna()].index.tolist()
            df[col] = df[col].fillna(median_val)
            log = CleaningLog(
                dataset_id=dataset_id,
                action="impute",
                details={"column": col, "rows": imputed_rows, "method": "median", "value": median_val},
            )
            db.add(log)
    db.commit()
    return df


def detect_suspicious(df: pd.DataFrame, dataset_id: str, db: Session) -> List[Dict[str, Any]]:
    """Flag numeric values that are >3 standard deviations from the mean.
    Returns list of flagged entries and logs them.
    """
    flags = []
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        series = df[col]
        mean = series.mean()
        std = series.std()
        if std == 0 or pd.isna(std):
            continue
        outlier_mask = (np.abs(series - mean) > 3 * std)
        outlier_idxs = series[outlier_mask].index.tolist()
        if outlier_idxs:
            flags.append({"column": col, "rows": outlier_idxs})
            log = CleaningLog(
                dataset_id=dataset_id,
                action="suspicious_flag",
                details={"column": col, "rows": outlier_idxs, "mean": mean, "std": std},
            )
            db.add(log)
    db.commit()
    return flags


def compute_quality_score(df: pd.DataFrame, dup_info: Dict[str, Any], suspicious: List[Dict[str, Any]]) -> float:
    """Simple composite score (0‑100).
    - Completeness: proportion of non‑missing cells.
    - Uniqueness: 1 - (duplicate rows / total rows).
    - Validity: 1 - (suspicious rows / total rows).
    The final score is the average of these three percentages.
    """
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()
    completeness = (1 - missing_cells / total_cells) * 100 if total_cells else 100
    total_rows = df.shape[0]
    uniqueness = (1 - dup_info.get("duplicate_count", 0) / total_rows) * 100 if total_rows else 100
    suspicious_rows = sum(len(item["rows"]) for item in suspicious)
    validity = (1 - suspicious_rows / total_rows) * 100 if total_rows else 100
    return round((completeness + uniqueness + validity) / 3, 2)
