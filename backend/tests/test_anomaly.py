"""
test_anomaly.py — Phase 6 tests
Validates feature engineering shape, model training/inference,
concrete anomaly explanation logic, and controlled evaluation set.
"""

import os
import shutil
import tempfile
import pandas as pd
import pytest

from backend.app.core.ev_features import engineer_features, FEATURE_COLUMNS
from backend.app.core.anomaly import (
    train_isolation_forest,
    predict_anomalies,
    compute_feature_stats,
    explain_anomaly,
    MODEL_ROOT,
)
from backend.app.core.anomaly_eval import (
    load_source_ev_data,
    build_synthetic_evaluation_set,
    run_evaluation,
)


@pytest.fixture
def sample_ev_df():
    return load_source_ev_data().head(60)


def test_feature_engineering_shape(sample_ev_df):
    enriched_df, feature_df = engineer_features(sample_ev_df)
    assert len(feature_df) == len(sample_ev_df)
    assert list(feature_df.columns) == FEATURE_COLUMNS
    assert not feature_df.isna().any().any(), "Feature matrix must not contain any NaNs"


def test_anomaly_training_and_inference(sample_ev_df):
    test_dataset_id = "test_dataset_anomaly_unit"
    model_path = train_isolation_forest(
        sample_ev_df,
        dataset_id=test_dataset_id,
        db_session=None,
        contamination=0.15,
    )

    try:
        assert os.path.exists(model_path)

        # Run inference
        anomalies = predict_anomalies(sample_ev_df, model_path)
        assert isinstance(anomalies, list)

        if anomalies:
            first = anomalies[0]
            assert "row_index" in first
            assert "anomaly_score" in first
            assert "explanation" in first
            assert isinstance(first["explanation"], str)
            assert "deviating_features" in first
    finally:
        if os.path.exists(model_path):
            os.remove(model_path)


def test_anomaly_explanation_logic():
    # Synthetic baseline stats
    stats = {
        "charging_duration_hours": {
            "mean": 2.5,
            "std": 0.8,
            "median": 2.4,
            "min": 0.5,
            "max": 6.0,
        },
        "observed_charging_rate_kw": {
            "mean": 30.0,
            "std": 5.0,
            "median": 29.0,
            "min": 10.0,
            "max": 50.0,
        },
    }

    anomalous_row = pd.Series({
        "charging_duration_hours": 48.0,  # ~56 std devs above mean
        "observed_charging_rate_kw": 31.0,  # normal
    })

    explanation = explain_anomaly(anomalous_row, stats)
    assert "charging_duration_hours" in explanation["explanation"]
    assert "higher than normal" in explanation["explanation"]
    assert len(explanation["deviating_features"]) >= 1
    assert explanation["deviating_features"][0]["feature"] == "charging_duration_hours"


def test_controlled_evaluation_pipeline(sample_ev_df):
    eval_set = build_synthetic_evaluation_set(sample_ev_df, n_normal=30)
    assert "is_anomaly" in eval_set.columns
    assert "anomaly_type" in eval_set.columns

    # Check there are both normal and injected anomalies
    assert (eval_set["is_anomaly"] == 0).sum() > 0
    assert (eval_set["is_anomaly"] == 1).sum() > 0

    results = run_evaluation(eval_df=eval_set, save_artifacts=False)
    assert "metrics" in results
    assert "precision" in results["metrics"]
    assert "recall" in results["metrics"]
    assert "f1_score" in results["metrics"]
    assert "confusion_matrix" in results
