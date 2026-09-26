"""
Phase 5 & 7 API endpoints — EV Mode:
Validation, Cleaning, Feature Engineering, Analytics, Visualizations, and Anomaly Explanations.
"""

from io import StringIO
import logging
import os
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.db.models import Dataset
from backend.app.core.ev_validation import run_ev_validation
from backend.app.core.ev_cleaning import clean_ev_dataset
from backend.app.core.ev_features import engineer_features, FEATURE_COLUMNS
from backend.app.core.anomaly import predict_anomalies, train_isolation_forest, MODEL_ROOT
from backend.app.core.ev_analytics import (
    assemble_ev_analytics,
    generate_ev_markdown_report,
)

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


@router.get("/{dataset_id}/ev/analytics", response_model=dict)
def get_ev_analytics(dataset_id: str, db: Session = Depends(get_db)):
    """Phase 7: Return comprehensive EV analytics and visualization payloads."""
    ds, df = _load_ev_dataset(dataset_id, db)
    payload = assemble_ev_analytics(df)
    return {
        "dataset_id": dataset_id,
        "filename": ds.filename,
        "analytics": payload,
    }


@router.get("/{dataset_id}/ev/anomalies", response_model=dict)
def get_ev_anomalies_with_explanations(dataset_id: str, db: Session = Depends(get_db)):
    """Phase 7: Retrieve all detected anomalies with concrete per-record explanations

    and summary distributions across charger types and vehicle models.
    """
    ds, df = _load_ev_dataset(dataset_id, db)

    model_path = MODEL_ROOT / f"{dataset_id}_iforest.pkl"
    # If model is not yet trained for this dataset, auto-train on its features
    if not model_path.exists():
        logger.info(f"Model not found for {dataset_id}, auto-training IsolationForest...")
        train_isolation_forest(df, dataset_id, db)

    raw_anomalies = predict_anomalies(df, str(model_path))

    # Enrich anomalies with context attributes from the original raw rows
    enriched_anomalies = []
    charger_distribution: dict[str, int] = {}
    vehicle_distribution: dict[str, int] = {}

    for anom in raw_anomalies:
        row_idx = anom["row_index"]
        row_data = df.iloc[row_idx].to_dict() if row_idx < len(df) else {}

        charger = str(row_data.get("Charger Type", "Unknown"))
        vehicle = str(row_data.get("Vehicle Model", "Unknown"))
        charger_distribution[charger] = charger_distribution.get(charger, 0) + 1
        vehicle_distribution[vehicle] = vehicle_distribution.get(vehicle, 0) + 1

        enriched_anomalies.append({
            "row_index": row_idx,
            "anomaly_score": anom["anomaly_score"],
            "explanation": anom["explanation"],
            "deviating_features": anom["deviating_features"],
            "vehicle_model": vehicle,
            "charger_type": charger,
            "energy_consumed_kwh": row_data.get("Energy Consumed (kWh)"),
            "charging_duration_hours": row_data.get("Charging Duration (hours)"),
            "charging_rate_kw": row_data.get("Charging Rate (kW)"),
        })

    anomaly_count = len(enriched_anomalies)
    total_records = len(df)
    anomaly_pct = round((anomaly_count / total_records) * 100, 2) if total_records > 0 else 0.0

    return {
        "dataset_id": dataset_id,
        "total_records": total_records,
        "anomaly_count": anomaly_count,
        "anomaly_percentage": anomaly_pct,
        "anomalies": enriched_anomalies,
        "anomaly_distribution": {
            "by_charger_type": charger_distribution,
            "by_vehicle_model": vehicle_distribution,
        },
    }


@router.get("/{dataset_id}/ev/report", response_class=StreamingResponse)
def get_ev_report(dataset_id: str, db: Session = Depends(get_db)):
    """Phase 7: Download executive EV telemetry report in Markdown format."""
    ds, df = _load_ev_dataset(dataset_id, db)
    report_content = generate_ev_markdown_report(df, dataset_id, ds.filename)
    stream = StringIO(report_content)
    return StreamingResponse(
        stream,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="ev_report_{dataset_id}.md"'},
    )
