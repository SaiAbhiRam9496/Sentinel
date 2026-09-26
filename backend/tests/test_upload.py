import io
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_upload_valid_csv():
    csv_content = "id,name,value\n1,Alice,10.5\n2,Bob,20.0\n3,Charlie,15.2\n"
    files = {"file": ("test_data.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    
    response = client.post("/api/datasets/upload", files=files)
    assert response.status_code == 201
    data = response.json()
    assert "dataset" in data
    dataset = data["dataset"]
    assert dataset["filename"] == "test_data.csv"
    assert dataset["row_count"] == 3
    assert dataset["column_count"] == 3
    assert dataset["columns"] == ["id", "name", "value"]
    assert dataset["detected_mode"] == "generic"


    # Test retrieval by ID
    dataset_id = dataset["id"]
    get_res = client.get(f"/api/datasets/{dataset_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == dataset_id
    assert get_res.json()["filename"] == "test_data.csv"

def test_list_datasets():
    response = client.get("/api/datasets")
    assert response.status_code == 200
    datasets = response.json()
    assert isinstance(datasets, list)
    assert len(datasets) >= 1

def test_upload_non_csv_rejected():
    files = {"file": ("notes.txt", io.BytesIO(b"Hello world"), "text/plain")}
    response = client.post("/api/datasets/upload", files=files)
    assert response.status_code == 400
    assert "Only CSV files" in response.json()["detail"]

def test_upload_empty_file_rejected():
    files = {"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
    response = client.post("/api/datasets/upload", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

def test_upload_whitespace_only_rejected():
    files = {"file": ("blank.csv", io.BytesIO(b"   \n\n  \t "), "text/csv")}
    response = client.post("/api/datasets/upload", files=files)
    assert response.status_code == 400
    assert "whitespace" in response.json()["detail"].lower()

def test_upload_headers_only_rejected():
    files = {"file": ("headers.csv", io.BytesIO(b"col1,col2,col3\n"), "text/csv")}
    response = client.post("/api/datasets/upload", files=files)
    assert response.status_code == 400
    assert "zero data rows" in response.json()["detail"].lower()

def test_get_nonexistent_dataset():
    response = client.get("/api/datasets/non-existent-uuid")
    assert response.status_code == 404
