"""
anomaly.py — Phase 6 API endpoints
Provides training and inference for the IsolationForest anomaly detection pipeline.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import os

from backend.app.db.session import get_db
from backend.app.db.models import Dataset
from backend.app.core.anomaly import train_isolation_forest, predict_anomalies

logger = logging.getLogger("sentinel.anomaly")
router = APIRouter(prefix="/datasets", tags=["Anomaly Detection"])


def _load_dataset(dataset_id: str, db: Session) -> Dataset:
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    if ds.detected_mode != "ev" and ds.detected_mode != "generic":
        # allow both modes for training
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dataset mode '{ds.detected_mode}' not suitable for anomaly pipeline",
        )
    return ds


@router.post("/{dataset_id}/anomaly/train", response_model=dict)
def train_anomaly_model(dataset_id: str, db: Session = Depends(get_db)):
    """Train IsolationForest model on the dataset's engineered features.
    Model is persisted to the filesystem under backend/models/anomaly/.
    """
    ds = _load_dataset(dataset_id, db)
    try:
        df = pd.read_csv(ds.file_path)
    except Exception as e:
        logger.error(f"Failed to load CSV for dataset {dataset_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to read dataset file.")
    model_path = train_isolation_forest(df, dataset_id, db)
    return {"dataset_id": dataset_id, "model_path": model_path, "status": "trained"}


@router.post("/{dataset_id}/anomaly/predict", response_model=dict)
def predict_anomalies_endpoint(dataset_id: str, db: Session = Depends(get_db)):
    """Run anomaly detection on the dataset using a previously trained model.
    Returns a list of rows flagged as anomalies with their scores.
    """
    ds = _load_dataset(dataset_id, db)
    model_path = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "anomaly")),
        f"{dataset_id}_iforest.pkl",
    )
    if not os.path.exists(model_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not trained for this dataset.")
    try:
        df = pd.read_csv(ds.file_path)
    except Exception as e:
        logger.error(f"Failed to load CSV for dataset {dataset_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to read dataset file.")
    anomalies = predict_anomalies(df, model_path)
    return {"dataset_id": dataset_id, "anomalies": anomalies, "count": len(anomalies)}
