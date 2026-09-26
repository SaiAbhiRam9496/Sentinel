"""
Phase 5 API endpoints — EV Mode: Validation, Cleaning, Feature Engineering
"""

import logging
import os
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.db.models import Dataset
from backend.app.core.ev_validation import run_ev_validation
from backend.app.core.ev_cleaning import clean_ev_dataset
from backend.app.core.ev_features import engineer_features, FEATURE_COLUMNS

logger = logging.getLogger("sentinel.ev")
router = APIRouter(prefix="/datasets", tags=["EV Mode"])


def _load_ev_dataset(dataset_id: str, db: Session) -> tuple[Dataset, pd.DataFrame]:
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    if ds.detected_mode != "ev":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dataset '{dataset_id}' is in '{ds.detected_mode}' mode. EV endpoints require mode=ev.",
        )
    try:
        df = pd.read_csv(ds.file_path)
    except Exception as e:
        logger.error(f"Failed to load dataset CSV {dataset_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to read dataset file.")
    return ds, df


@router.post("/{dataset_id}/ev/validate", response_model=dict)
def ev_validate(dataset_id: str, db: Session = Depends(get_db)):
    """Run EV-specific rule-based validity checks. Returns violations list."""
    ds, df = _load_ev_dataset(dataset_id, db)
    result = run_ev_validation(df, dataset_id, db)
    return {"dataset_id": dataset_id, "validation": result}


@router.post("/{dataset_id}/ev/clean", response_model=dict)
def ev_clean(dataset_id: str, db: Session = Depends(get_db)):
    """Apply EV cleaning pipeline (drop missing core fields, median-impute numeric gaps).
    Saves cleaned file back to disk and updates dataset metadata."""
    ds, df = _load_ev_dataset(dataset_id, db)
    cleaned_df, rows_dropped = clean_ev_dataset(df, dataset_id, db)

    cleaned_df.to_csv(ds.file_path, index=False)
    ds.row_count = len(cleaned_df)
    ds.column_count = len(cleaned_df.columns)
    db.commit()

    return {
        "dataset_id": dataset_id,
        "original_row_count": len(df),
        "cleaned_row_count": len(cleaned_df),
        "rows_dropped": rows_dropped,
        "message": "EV dataset cleaned successfully",
    }


@router.post("/{dataset_id}/ev/features", response_model=dict)
def ev_features(dataset_id: str, db: Session = Depends(get_db)):
    """Run feature engineering on the EV dataset.
    Returns the list of engineered feature columns and sample stats.
    Enriched dataset is saved back to disk for use by the ML pipeline."""
    ds, df = _load_ev_dataset(dataset_id, db)
    enriched_df, feature_df = engineer_features(df)

    # Save enriched CSV (adds derived feature columns) back to disk
    enriched_df.to_csv(ds.file_path, index=False)
    ds.column_count = len(enriched_df.columns)
    db.commit()

    # Return column names and basic numeric summary
    feature_summary = {}
    for col in FEATURE_COLUMNS:
        series = feature_df[col]
        feature_summary[col] = {
            "mean": round(series.mean(), 4),
            "std": round(series.std(), 4),
            "min": round(series.min(), 4),
            "max": round(series.max(), 4),
        }

    return {
        "dataset_id": dataset_id,
        "feature_columns": FEATURE_COLUMNS,
        "feature_summary": feature_summary,
    }
