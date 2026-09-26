"""
anomaly.py — Phase 6
ML Anomaly Detection Pipeline using IsolationForest.
Provides training, inference, and concrete anomaly explanation per flagged record.
Persists model and feature distribution statistics as a versioned artifact.
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from backend.app.core.ev_features import engineer_features, FEATURE_COLUMNS
from backend.app.db.models import CleaningLog

# Directory to store trained models per dataset
MODEL_ROOT = Path(__file__).resolve().parent.parent.parent / "models" / "anomaly"
MODEL_ROOT.mkdir(parents=True, exist_ok=True)


def _log(db_session, dataset_id: str, action: str, details: Dict[str, Any]) -> None:
    if db_session is not None:
        db_session.add(CleaningLog(dataset_id=dataset_id, action=action, details=details))
        db_session.commit()


def compute_feature_stats(feature_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute baseline distribution statistics for each feature used in anomaly detection."""
    stats = {}
    for col in feature_df.columns:
        series = pd.to_numeric(feature_df[col], errors="coerce").dropna()
        mean_val = float(series.mean()) if len(series) > 0 else 0.0
        std_val = float(series.std()) if len(series) > 1 else 1.0
        if std_val == 0.0 or np.isnan(std_val):
            std_val = 1.0
        stats[col] = {
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "median": round(float(series.median()), 4) if len(series) > 0 else 0.0,
            "min": round(float(series.min()), 4) if len(series) > 0 else 0.0,
            "max": round(float(series.max()), 4) if len(series) > 0 else 0.0,
            "p05": round(float(series.quantile(0.05)), 4) if len(series) > 0 else 0.0,
            "p95": round(float(series.quantile(0.95)), 4) if len(series) > 0 else 0.0,
        }
    return stats


def train_isolation_forest(
    df: pd.DataFrame,
    dataset_id: str,
    db_session=None,
    model_version: str = "1.0.0",
    contamination: float | str = "auto",
) -> str:
    """Train an IsolationForest on the engineered feature matrix.
    Saves model bundle including training distribution statistics for anomaly explanations.
    Returns the absolute path to the saved model pickle.
    """
    _enriched, feature_df = engineer_features(df)

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
    )
    model.fit(feature_df)

    stats = compute_feature_stats(feature_df)

    artifact = {
        "model": model,
        "version": model_version,
        "feature_names": FEATURE_COLUMNS,
        "feature_stats": stats,
        "n_samples": int(len(feature_df)),
    }

    model_path = MODEL_ROOT / f"{dataset_id}_iforest.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    _log(
        db_session,
        dataset_id,
        "anomaly_training",
        {
            "model_path": str(model_path),
            "n_features": feature_df.shape[1],
            "n_samples": len(feature_df),
            "version": model_version,
        },
    )
    return str(model_path)


def explain_anomaly(
    row_features: pd.Series,
    feature_stats: Dict[str, Dict[str, float]],
    top_k: int = 3,
) -> Dict[str, Any]:
    """Identify which features deviate most from the baseline distribution and generate

    a human-readable explanation with concrete numbers.
    """
    deviations = []

    for feat_name, stats in feature_stats.items():
        if feat_name not in row_features.index:
            continue
        val = row_features[feat_name]
        if pd.isna(val):
            continue

        mean = stats["mean"]
        std = stats["std"]
        z_score = (val - mean) / (std if std > 0 else 1.0)
        abs_z = abs(z_score)

        direction = "higher" if z_score > 0 else "lower"
        deviations.append({
            "feature": feat_name,
            "value": round(float(val), 2),
            "mean": round(float(mean), 2),
            "std": round(float(std), 2),
            "z_score": round(float(z_score), 2),
            "abs_z": round(float(abs_z), 2),
            "direction": direction,
            "normal_range": [stats.get("p05", mean - 2 * std), stats.get("p95", mean + 2 * std)],
        })

    # Sort descending by absolute z-score
    deviations.sort(key=lambda x: x["abs_z"], reverse=True)

    # Pick top drivers (deviations with |z| >= 1.5, or top 1 if none reach threshold)
    top_drivers = [d for d in deviations if d["abs_z"] >= 1.5][:top_k]
    if not top_drivers and deviations:
        top_drivers = deviations[:1]

    # Build concise readable explanation
    reasons = []
    for d in top_drivers:
        reasons.append(
            f"{d['feature']} is {d['direction']} than normal ({d['value']} vs avg {d['mean']} ± {d['std']})"
        )

    explanation = "; ".join(reasons) if reasons else "Unusual combination of feature values"

    return {
        "explanation": explanation,
        "deviating_features": [
            {
                "feature": d["feature"],
                "value": d["value"],
                "mean": d["mean"],
                "std": d["std"],
                "z_score": d["z_score"],
                "direction": d["direction"],
            }
            for d in top_drivers
        ],
    }


def predict_anomalies(
    df: pd.DataFrame,
    model_path: str,
) -> List[Dict[str, Any]]:
    """Run inference using a previously saved IsolationForest model artifact.
    Returns a list of anomalies with row index, anomaly score, and explanation.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    if isinstance(artifact, dict) and "model" in artifact:
        model: IsolationForest = artifact["model"]
        feature_stats = artifact.get("feature_stats", {})
    else:
        model = artifact
        feature_stats = {}

    _, feature_df = engineer_features(df)
    scores = model.decision_function(feature_df)  # higher => more normal
    preds = model.predict(feature_df)  # -1 = anomaly, 1 = normal

    # If feature_stats weren't saved with model, compute on current data
    if not feature_stats:
        feature_stats = compute_feature_stats(feature_df)

    anomalies = []
    for idx, (pred, score) in enumerate(zip(preds, scores)):
        if pred == -1:
            row_feats = feature_df.iloc[idx]
            exp_data = explain_anomaly(row_feats, feature_stats)
            anomalies.append({
                "row_index": int(idx),
                "anomaly_score": round(float(score), 4),
                "explanation": exp_data["explanation"],
                "deviating_features": exp_data["deviating_features"],
            })

    return anomalies
