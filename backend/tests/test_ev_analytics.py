"""
test_ev_analytics.py — Phase 7 tests
Validates EV KPIs, charger breakdown, vehicle model analysis, time-of-day demand,
temperature impacts, markdown report generation, and EV analytics API endpoints.
"""

import io
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.ev_analytics import (
    compute_ev_kpis,
    compute_charger_type_analysis,
    compute_vehicle_model_analysis,
    compute_time_of_day_analysis,
    compute_temperature_impact,
    compute_ev_distributions,
    generate_ev_markdown_report,
    assemble_ev_analytics,
)
from backend.app.core.anomaly_eval import load_source_ev_data


@pytest.fixture
def ev_df():
    return load_source_ev_data().head(100)


def test_ev_kpis(ev_df):
    kpis = compute_ev_kpis(ev_df)
    assert kpis["total_sessions"] == 100
    assert kpis["total_energy_kwh"] > 0
    assert kpis["avg_charging_rate_kw"] > 0
    assert kpis["avg_duration_hours"] > 0


def test_charger_type_analysis(ev_df):
    chargers = compute_charger_type_analysis(ev_df)
    assert len(chargers) > 0
    first = chargers[0]
    assert "charger_type" in first
    assert "session_count" in first
    assert "total_energy_kwh" in first
    assert "avg_charging_rate_kw" in first
    # Total percentage should sum to approximately 100%
    total_pct = sum(c["session_share_pct"] for c in chargers)
    assert 99.0 <= total_pct <= 101.0


def test_vehicle_model_analysis(ev_df):
    models = compute_vehicle_model_analysis(ev_df)
    assert len(models) > 0
    first = models[0]
    assert "vehicle_model" in first
    assert "avg_battery_capacity_kwh" in first
    assert "avg_energy_consumed_kwh" in first


def test_time_of_day_and_temp(ev_df):
    tod = compute_time_of_day_analysis(ev_df)
    assert "by_period" in tod
    assert "by_hour" in tod
    if tod["by_hour"]:
        assert len(tod["by_hour"]) == 24

    temps = compute_temperature_impact(ev_df)
    assert isinstance(temps, list)


def test_ev_distributions(ev_df):
    dists = compute_ev_distributions(ev_df)
    assert "energy_consumed_kwh" in dists
    assert "charging_duration_hours" in dists
    assert len(dists["energy_consumed_kwh"]) > 0


def test_ev_markdown_report(ev_df):
    report = generate_ev_markdown_report(ev_df, "test-ds-123", "ev_sample.csv")
    assert "# Sentinel — EV Telemetry & Charging Analytics Report" in report
    assert "Executive KPIs" in report
    assert "Infrastructure & Charger Performance" in report
    assert "Fleet Vehicle Model Analysis" in report


def test_api_ev_analytics_flow(ev_df):
    client = TestClient(app)

    # 1. Upload sample EV dataset
    csv_bytes = ev_df.to_csv(index=False).encode("utf-8")
    files = {"file": ("test_ev_flow.csv", io.BytesIO(csv_bytes), "text/csv")}
    upload_res = client.post("/api/datasets/upload", files=files)
    assert upload_res.status_code == 201
    dataset_id = upload_res.json()["dataset"]["id"]

    try:
        # 2. Query EV analytics endpoint
        analytics_res = client.get(f"/api/datasets/{dataset_id}/ev/analytics")
        assert analytics_res.status_code == 200
        data = analytics_res.json()
        assert data["dataset_id"] == dataset_id
        assert "analytics" in data
        assert "kpis" in data["analytics"]
        assert "charger_types" in data["analytics"]

        # 3. Query EV anomalies with explanations endpoint
        anomalies_res = client.get(f"/api/datasets/{dataset_id}/ev/anomalies")
        assert anomalies_res.status_code == 200
        anom_data = anomalies_res.json()
        assert "anomaly_count" in anom_data
        assert "anomalies" in anom_data
        assert "anomaly_distribution" in anom_data

        # 4. Query EV report download
        report_res = client.get(f"/api/datasets/{dataset_id}/ev/report")
        assert report_res.status_code == 200
        assert "Sentinel" in report_res.text
    finally:
        # Cleanup uploaded test file if exists
        upload_path = upload_res.json()["dataset"].get("file_path")
        if upload_path and os.path.exists(upload_path):
            os.remove(upload_path)
