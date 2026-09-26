"""
anomaly_eval.py — Phase 6
Controlled Evaluation Pipeline for EV ML Anomaly Detection.

Builds a separate, clearly-labeled synthetic evaluation dataset with injected anomalies
(implausible charging rates, impossible SOC jumps, battery capacity violations, phantom stalls).
Evaluates IsolationForest detection performance: precision, recall, F1, and confusion matrix.
Results are saved to backend/data/eval/eval_results.json.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from backend.app.core.ev_features import engineer_features, FEATURE_COLUMNS
from backend.app.core.anomaly import compute_feature_stats, explain_anomaly

logger = logging.getLogger("sentinel.anomaly_eval")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
EVAL_DIR = DATA_DIR / "eval"
SAMPLE_DIR = DATA_DIR / "sample"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def load_source_ev_data() -> pd.DataFrame:
    """Load ev_charging_patterns.csv from backend/data/sample or fallback paths."""
    candidates = [
        SAMPLE_DIR / "ev_charging_patterns.csv",
        Path(__file__).resolve().parent.parent.parent.parent / "Datasets" / "ev_charging_patterns.csv",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(f"Could not find ev_charging_patterns.csv in candidates: {candidates}")


def build_synthetic_evaluation_set(
    source_df: pd.DataFrame,
    n_normal: int = 150,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Create a controlled evaluation dataset containing verified normal records

    and deliberately injected synthetic anomaly records.

    Label conventions:
      - is_anomaly: 0 (normal), 1 (anomaly)
      - anomaly_type: description of the injected defect or 'normal'
    """
    rng = np.random.RandomState(random_seed)

    # 1. Select clean normal rows
    valid_mask = (
        (source_df["State of Charge (Start %)"] >= 0)
        & (source_df["State of Charge (End %)"] <= 100)
        & (source_df["State of Charge (End %)"] >= source_df["State of Charge (Start %)"])
        & (source_df["Energy Consumed (kWh)"] > 0)
        & (source_df["Charging Duration (hours)"] > 0)
        & (source_df["Charging Duration (hours)"] < 12)
    )
    clean_pool = source_df[valid_mask].copy()
    if len(clean_pool) < n_normal:
        clean_pool = source_df.copy()

    sampled_normal = clean_pool.sample(n=min(n_normal, len(clean_pool)), random_state=random_seed).copy()
    sampled_normal["is_anomaly"] = 0
    sampled_normal["anomaly_type"] = "normal"

    # 2. Inject deliberate synthetic anomalies into copies
    anomaly_rows = []

    # Category A: Massive power surge / implausible charging rate
    # E.g. 500-900 kW on a standard Level 2 or Level 1 charger
    for _, base_row in clean_pool.sample(n=8, random_state=random_seed + 1).iterrows():
        row = base_row.copy()
        row["Charging Rate (kW)"] = float(rng.uniform(600.0, 950.0))
        row["Energy Consumed (kWh)"] = float(rng.uniform(300.0, 500.0))
        row["Charging Duration (hours)"] = float(rng.uniform(0.1, 0.3))
        row["is_anomaly"] = 1
        row["anomaly_type"] = "power_surge_implausible_rate"
        anomaly_rows.append(row)

    # Category B: Impossible SOC delta (severe battery reversal during charge)
    for _, base_row in clean_pool.sample(n=8, random_state=random_seed + 2).iterrows():
        row = base_row.copy()
        row["State of Charge (Start %)"] = float(rng.uniform(85.0, 98.0))
        row["State of Charge (End %)"] = float(rng.uniform(5.0, 15.0))  # Lost 80% SOC while charging
        row["Energy Consumed (kWh)"] = float(rng.uniform(40.0, 80.0))
        row["is_anomaly"] = 1
        row["anomaly_type"] = "inverted_soc_drain"
        anomaly_rows.append(row)

    # Category C: Energy consumed far exceeds physical battery capacity
    for _, base_row in clean_pool.sample(n=8, random_state=random_seed + 3).iterrows():
        row = base_row.copy()
        batt = float(row.get("Battery Capacity (kWh)", 50.0))
        row["Energy Consumed (kWh)"] = batt * float(rng.uniform(5.0, 9.0))  # 500-900% capacity
        row["Charging Duration (hours)"] = float(rng.uniform(1.0, 2.5))
        row["is_anomaly"] = 1
        row["anomaly_type"] = "energy_exceeds_capacity"
        anomaly_rows.append(row)

    # Category D: Stalled phantom charging session (huge duration, zero/negligible energy)
    for _, base_row in clean_pool.sample(n=6, random_state=random_seed + 4).iterrows():
        row = base_row.copy()
        row["Charging Duration (hours)"] = float(rng.uniform(48.0, 120.0))  # 2 to 5 days
        row["Energy Consumed (kWh)"] = float(rng.uniform(0.05, 0.3))
        row["Charging Rate (kW)"] = 0.05
        row["is_anomaly"] = 1
        row["anomaly_type"] = "stalled_phantom_session"
        anomaly_rows.append(row)

    # Category E: Extreme thermal & telemetry outlier
    for _, base_row in clean_pool.sample(n=5, random_state=random_seed + 5).iterrows():
        row = base_row.copy()
        row["Temperature (°C)"] = float(rng.choice([85.0, 92.0, -45.0]))
        row["Distance Driven (since last charge) (km)"] = float(rng.uniform(3500.0, 6000.0))
        row["is_anomaly"] = 1
        row["anomaly_type"] = "extreme_temperature_telemetry"
        anomaly_rows.append(row)

    anomaly_df = pd.DataFrame(anomaly_rows)

    # Combine into unified controlled set, shuffle rows
    eval_df = pd.concat([sampled_normal, anomaly_df], ignore_index=True)
    eval_df = eval_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    return eval_df


def run_evaluation(
    eval_df: Optional[pd.DataFrame] = None,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """Train IsolationForest on full dataset, run inference on the controlled

    synthetic evaluation set, and compute standard classification metrics.
    """
    raw_df = load_source_ev_data()

    if eval_df is None:
        eval_df = build_synthetic_evaluation_set(raw_df)

    # 1. Train clean IsolationForest baseline model on source data
    _, train_features = engineer_features(raw_df)
    model = IsolationForest(n_estimators=100, contamination=0.10, random_state=42)
    model.fit(train_features)
    feature_stats = compute_feature_stats(train_features)

    # 2. Extract features from evaluation set (exclude evaluation metadata columns)
    feature_eval_input = eval_df.drop(columns=["is_anomaly", "anomaly_type"], errors="ignore")
    _, eval_features = engineer_features(feature_eval_input)

    # 3. Model predictions: IsolationForest outputs -1 (anomaly) and 1 (normal)
    raw_preds = model.predict(eval_features)
    scores = model.decision_function(eval_features)
    pred_labels = np.where(raw_preds == -1, 1, 0)
    actual_labels = eval_df["is_anomaly"].values

    # 4. Compute metrics
    tp = int(np.sum((pred_labels == 1) & (actual_labels == 1)))
    fp = int(np.sum((pred_labels == 1) & (actual_labels == 0)))
    tn = int(np.sum((pred_labels == 0) & (actual_labels == 0)))
    fn = int(np.sum((pred_labels == 0) & (actual_labels == 1)))

    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1 = round(2 * (precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0
    accuracy = round((tp + tn) / len(actual_labels), 4) if len(actual_labels) > 0 else 0.0

    # Recall breakdown per injected category
    category_breakdown = {}
    for cat in eval_df["anomaly_type"].unique():
        if cat == "normal":
            continue
        mask = eval_df["anomaly_type"] == cat
        cat_total = int(np.sum(mask))
        cat_detected = int(np.sum((pred_labels == 1) & mask))
        category_breakdown[cat] = {
            "total_injected": cat_total,
            "detected": cat_detected,
            "recall": round(cat_detected / cat_total, 4) if cat_total > 0 else 0.0,
        }

    results = {
        "model": "IsolationForest",
        "n_estimators": 100,
        "contamination": 0.10,
        "feature_set": FEATURE_COLUMNS,
        "evaluation_dataset_size": len(eval_df),
        "actual_normals": int(np.sum(actual_labels == 0)),
        "actual_anomalies": int(np.sum(actual_labels == 1)),
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
        },
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        },
        "per_category_performance": category_breakdown,
    }

    if save_artifacts:
        eval_csv_path = EVAL_DIR / "synthetic_anomalies_eval.csv"
        eval_df.to_csv(eval_csv_path, index=False)

        metrics_path = EVAL_DIR / "eval_results.json"
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Evaluation dataset saved to {eval_csv_path}")
        logger.info(f"Evaluation metrics saved to {metrics_path}")

    return results


if __name__ == "__main__":
    print("Running controlled anomaly evaluation...")
    metrics = run_evaluation(save_artifacts=True)
    print(json.dumps(metrics, indent=2))
