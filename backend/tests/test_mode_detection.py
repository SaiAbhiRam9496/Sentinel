import os
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.schema import (
    infer_column_type,
    detect_schema_and_types,
    detect_mode,
    CANONICAL_EV_COLUMNS
)

client = TestClient(app)
DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "Datasets"))

def test_column_type_inference():
    df = pd.DataFrame({
        "int_col": [1, 2, 3, 4],
        "float_col": [1.1, 2.5, 3.0, 4.8],
        "bool_col": [True, False, True, False],
        "date_col": ["2024-01-01 10:00:00", "2024-01-01 11:00:00", "2024-01-01 12:00:00", "2024-01-01 13:00:00"],
        "category_col": ["Fast", "Slow", "Fast", "Slow"],
        "text_col": ["Unique text Alpha", "Unique text Beta", "Unique text Gamma", "Unique text Delta"]
    })

    types = detect_schema_and_types(df)
    assert types["int_col"]["type"] == "integer"
    assert types["float_col"]["type"] == "float"
    assert types["bool_col"]["type"] == "boolean"
    assert types["date_col"]["type"] == "datetime"
    assert types["category_col"]["type"] == "categorical"

def test_ev_charging_patterns_mode_detection():
    file_path = os.path.join(DATASETS_DIR, "ev_charging_patterns.csv")
    assert os.path.exists(file_path), f"Dataset not found at {file_path}"

    df = pd.read_csv(file_path)
    mode, reason, details = detect_mode(df)
    assert mode == "ev"
    assert details["matched_canonical_count"] == 20
    assert details["canonical_match_percentage"] == 100.0
    assert "EV Charging Telemetry detected" in reason

def test_station_data_dataverse_mode_detection():
    file_path = os.path.join(DATASETS_DIR, "station_data_dataverse.csv")
    assert os.path.exists(file_path), f"Dataset not found at {file_path}"

    df = pd.read_csv(file_path)
    mode, reason, details = detect_mode(df)
    assert mode == "generic"
    assert details["canonical_match_percentage"] < 75.0
    assert "Generic" in reason

def test_ev_stations_2025_mode_detection():
    file_path = os.path.join(DATASETS_DIR, "ev_stations_2025.csv")
    assert os.path.exists(file_path), f"Dataset not found at {file_path}"

    df = pd.read_csv(file_path)
    mode, reason, details = detect_mode(df)
    assert mode == "generic"
    assert details["canonical_match_percentage"] < 75.0
    assert "Generic" in reason

def test_api_upload_mode_detection_end_to_end():
    file_path = os.path.join(DATASETS_DIR, "ev_charging_patterns.csv")
    with open(file_path, "rb") as f:
        files = {"file": ("ev_charging_patterns.csv", f, "text/csv")}
        res = client.post("/api/datasets/upload", files=files)
    
    assert res.status_code == 201
    dataset = res.json()["dataset"]
    assert dataset["detected_mode"] == "ev"
    assert dataset["match_details"]["canonical_match_percentage"] == 100.0
    assert "schema_details" in dataset

    # Test explicit detect-mode endpoint
    detect_res = client.post(f"/api/datasets/{dataset['id']}/detect-mode")
    assert detect_res.status_code == 200
    assert detect_res.json()["detected_mode"] == "ev"
