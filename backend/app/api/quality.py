import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import pandas as pd

from backend.app.db.session import get_db
from backend.app.db.models import Dataset, CleaningLog
from backend.app.core.quality import (
    compute_profile,
    detect_duplicates,
    handle_missing_values,
    detect_suspicious,
    compute_quality_score,
)

logger = logging.getLogger("sentinel.quality")
router = APIRouter(prefix="/datasets", tags=["Data Quality"])

def _load_dataframe(dataset: Dataset) -> pd.DataFrame:
    try:
        df = pd.read_csv(dataset.file_path)
        return df
    except Exception as e:
        logger.error(f"Failed to read dataset CSV {dataset.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to load dataset for analysis.",
        )

@router.get("/{dataset_id}/profile", response_model=dict)
def get_profile(dataset_id: str, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    df = _load_dataframe(dataset)
    profile = compute_profile(df)
    duplicates = detect_duplicates(df)
    return {
        "dataset_id": dataset_id,
        "profile": profile,
        "duplicates": duplicates,
    }

@router.post("/{dataset_id}/clean", response_model=dict)
def clean_dataset(dataset_id: str, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    df = _load_dataframe(dataset)
    df = handle_missing_values(df, dataset_id, db)
    suspicious = detect_suspicious(df, dataset_id, db)
    df.to_csv(dataset.file_path, index=False)
    dataset.row_count = len(df)
    dataset.column_count = len(df.columns)
    db.commit()
    return {"message": "Cleaning complete", "suspicious_flags": suspicious}

@router.get("/{dataset_id}/quality-score", response_model=dict)
def get_quality_score(dataset_id: str, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    df = _load_dataframe(dataset)
    dup_info = detect_duplicates(df)
    suspicious = detect_suspicious(df, dataset_id, db)
    score = compute_quality_score(df, dup_info, suspicious)
    return {"dataset_id": dataset_id, "quality_score": score}


@router.get("/{dataset_id}/cleaning-log", response_model=dict)
def get_cleaning_log(dataset_id: str, db: Session = Depends(get_db)):
    """Return the full cleaning/transparency log for a dataset.
    Shows every drop, impute, and suspicious-flag action with reasons.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    logs = (
        db.query(CleaningLog)
        .filter(CleaningLog.dataset_id == dataset_id)
        .order_by(CleaningLog.created_at.asc())
        .all()
    )
    return {"dataset_id": dataset_id, "log": [l.to_dict() for l in logs]}
