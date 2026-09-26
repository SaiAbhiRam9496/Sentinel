import os
import io
import uuid
import pandas as pd
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status
from sqlalchemy.orm import Session
from backend.app.config import settings
from backend.app.db.session import get_db
from backend.app.db.models import Dataset
from backend.app.core.schema import detect_schema_and_types, detect_mode
import logging

logger = logging.getLogger("sentinel.upload")
router = APIRouter(prefix="/datasets", tags=["Datasets"])


UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # 1. Validate file extension
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type for '{filename}'. Only CSV files (.csv) are accepted."
        )

    # 2. Read file with size cap enforcement
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    content = bytearray()
    chunk_size = 1024 * 1024  # 1MB chunks

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        content.extend(chunk)
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum upload size of {settings.MAX_UPLOAD_SIZE_MB}MB."
            )

    file_size_bytes = len(content)

    # 3. Check for empty file
    if file_size_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded CSV file is empty (0 bytes)."
        )

    # 4. Handle decoding (UTF-8 with fallback to latin-1)
    decoded_text = None
    for encoding in ["utf-8-sig", "utf-8", "latin-1"]:
        try:
            decoded_text = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if decoded_text is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to decode file. Please upload a valid text-encoded CSV file."
        )

    # Check if file has only whitespace
    if not decoded_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file contains only empty whitespace."
        )

    # 5. Parse CSV structure with pandas
    try:
        df = pd.read_csv(io.StringIO(decoded_text))
    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file contains no data or could not be parsed."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Malformed CSV: Failed to parse tabular data. Details: {str(e)}"
        )

    # Check columns
    if len(df.columns) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The CSV file has no columns."
        )

    # Check headers-only (0 rows of data)
    if len(df) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded CSV contains column headers but zero data rows."
        )

    # 6. Schema and Mode Auto-Detection
    schema_details = detect_schema_and_types(df)
    detected_mode, mode_reason, match_details = detect_mode(df)

    # 7. Save file to disk
    dataset_id = str(uuid.uuid4())
    stored_filename = f"{dataset_id}.csv"
    stored_path = os.path.join(UPLOAD_DIR, stored_filename)

    with open(stored_path, "wb") as f:
        f.write(content)

    # 8. Record dataset in PostgreSQL
    dataset_record = Dataset(
        id=dataset_id,
        filename=filename,
        file_path=stored_path,
        file_size_bytes=file_size_bytes,
        row_count=len(df),
        column_count=len(df.columns),
        columns=list(df.columns),
        schema_details=schema_details,
        detected_mode=detected_mode,
        mode_reason=mode_reason,
        match_details=match_details,
        status="uploaded"
    )

    db.add(dataset_record)
    db.commit()
    db.refresh(dataset_record)

    logger.info(f"Dataset uploaded and analyzed: ID={dataset_id}, Filename={filename}, Mode={detected_mode}")

    return {
        "message": "Dataset uploaded and analyzed successfully",
        "dataset": dataset_record.to_dict()
    }

@router.post("/{dataset_id}/detect-mode", status_code=status.HTTP_200_OK)
def trigger_mode_detection(
    dataset_id: str,
    db: Session = Depends(get_db)
):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID '{dataset_id}' not found."
        )

    if not os.path.exists(dataset.file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Underlying dataset CSV file missing on server."
        )

    df = pd.read_csv(dataset.file_path)
    schema_details = detect_schema_and_types(df)
    detected_mode, mode_reason, match_details = detect_mode(df)

    dataset.schema_details = schema_details
    dataset.detected_mode = detected_mode
    dataset.mode_reason = mode_reason
    dataset.match_details = match_details
    db.commit()
    db.refresh(dataset)

    return {
        "dataset_id": dataset_id,
        "detected_mode": detected_mode,
        "mode_reason": mode_reason,
        "match_details": match_details,
        "schema_details": schema_details
    }


@router.get("/{dataset_id}", status_code=status.HTTP_200_OK)
def get_dataset_metadata(
    dataset_id: str,
    db: Session = Depends(get_db)
):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID '{dataset_id}' not found."
        )
    return dataset.to_dict()

@router.get("", status_code=status.HTTP_200_OK)
def list_datasets(
    limit: int = 20,
    db: Session = Depends(get_db)
):
    datasets = db.query(Dataset).order_by(Dataset.created_at.desc()).limit(limit).all()
    return [d.to_dict() for d in datasets]
