"""
anomaly.py — Phase 6
ML Anomaly Detection Pipeline using IsolationForest.
Provides training and inference utilities, persists model as a pickle.
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from sklearn.ensemble import IsolationForest

from backend.app.core.ev_features import engineer_features
from backend.app.db.models import CleaningLog

# Directory to store trained models per dataset
MODEL_ROOT = Path(__file__).resolve().parent.parent.parent / "models" / "anomaly"
MODEL_ROOT.mkdir(parents=True, exist_ok=True)


def _log(db_session, dataset_id: str, action: str, details: Dict[str, Any]) -> None:
    db_session.add(CleaningLog(dataset_id=dataset_id, action=action, details=details))
    db_session.commit()


def train_isolation_forest(df: pd.DataFrame, dataset_id: str, db_session) -> str:
    """Train an IsolationForest on the engineered feature matrix.
    Returns the absolute path to the saved model pickle.
    """
    # Engineer features – reuse EV feature pipeline (works for generic too)
    _enriched, feature_df = engineer_features(df)

    model = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
    model.fit(feature_df)

    model_path = MODEL_ROOT / f"{dataset_id}_iforest.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    _log(db_session, dataset_id, "anomaly_training", {"model_path": str(model_path), "n_features": feature_df.shape[1]})
    return str(model_path)


def predict_anomalies(df: pd.DataFrame, model_path: str) -> List[Dict[str, Any]]:
    """Run inference using a previously saved IsolationForest model.
    Returns a list of anomalies with row index and anomaly score.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, "rb") as f:
        model: IsolationForest = pickle.load(f)

    _, feature_df = engineer_features(df)
    scores = model.decision_function(feature_df)  # higher => more normal
    preds = model.predict(feature_df)  # -1 = anomaly, 1 = normal

    anomalies = []
    for idx, (pred, score) in enumerate(zip(preds, scores)):
        if pred == -1:
            anomalies.append({"row_index": idx, "anomaly_score": float(score)})
    return anomalies
