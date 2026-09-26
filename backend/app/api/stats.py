import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
import pandas as pd
from io import StringIO

from backend.app.db.session import get_db
from backend.app.db.models import Dataset
from backend.app.core.stats import generate_summary_stats, generate_histogram
from backend.app.core.quality import compute_profile, compute_quality_score, detect_duplicates, detect_suspicious

logger = logging.getLogger("sentinel.stats")
router = APIRouter(prefix="/datasets", tags=["Stats & Visualizations"])

def _load_dataframe(dataset: Dataset) -> pd.DataFrame:
    try:
        return pd.read_csv(dataset.file_path)
    except Exception as e:
        logger.error(f"Failed to load CSV for dataset {dataset.id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to read dataset file.")

@router.get("/{dataset_id}/summary-stats", response_model=dict)
def get_summary_stats(dataset_id: str, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    df = _load_dataframe(ds)
    return {"dataset_id": dataset_id, "summary_stats": generate_summary_stats(df)}

@router.get("/{dataset_id}/visualizations", response_model=dict)
def get_visualizations(dataset_id: str, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    df = _load_dataframe(ds)
    viz = {}
    for col in df.columns:
        ser = df[col]
        if pd.api.types.is_numeric_dtype(ser):
            viz[col] = {"type": "histogram", "bins": generate_histogram(ser)}
        else:
            top_counts = ser.value_counts().head(5).to_dict()
            viz[col] = {"type": "bar", "top_categories": top_counts}
    return {"dataset_id": dataset_id, "visualizations": viz}

@router.get("/{dataset_id}/download", response_class=FileResponse)
def download_dataset(dataset_id: str, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    return FileResponse(path=ds.file_path, filename=ds.filename, media_type="text/csv")

@router.get("/{dataset_id}/report", response_class=StreamingResponse)
def download_report(dataset_id: str, db: Session = Depends(get_db)):
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    df = _load_dataframe(ds)
    profile = compute_profile(df)
    dup_info = detect_duplicates(df)
    suspicious = detect_suspicious(df, dataset_id, db)
    quality_score = compute_quality_score(df, dup_info, suspicious)
    stats = generate_summary_stats(df)
    md_parts = ["# Dataset Report", f"**Dataset ID:** {dataset_id}", f"**Filename:** {ds.filename}", "\n## Profile", f"```json\n{profile}\n```", "\n## Summary Statistics", f"```json\n{stats}\n```", "\n## Data Quality Score", f"{quality_score}", "\n## Duplicate Info", f"```json\n{dup_info}\n```", "\n## Suspicious Flags", f"```json\n{suspicious}\n```"]
    markdown = "\n\n".join(md_parts)
    stream = StringIO(markdown)
    return StreamingResponse(iter([stream.getvalue()]), media_type="text/markdown", headers={"Content-Disposition": f"attachment; filename=report_{dataset_id}.md"})


@router.get("/{dataset_id}/data-dictionary", response_model=dict)
def get_data_dictionary(dataset_id: str, db: Session = Depends(get_db)):
    """Generate a plain-language data dictionary for each column."""
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    df = _load_dataframe(ds)
    dictionary = {}
    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        total = len(series)
        unique = int(series.nunique(dropna=True))
        if pd.api.types.is_numeric_dtype(series):
            col_type = "numeric"
            description = (
                f"Numeric column with {unique} unique values. "
                f"Range: {round(series.min(), 2)} – {round(series.max(), 2)}. "
                f"Mean: {round(series.mean(), 2)}. Missing: {missing}/{total}."
            )
        else:
            col_type = "categorical / text"
            top = series.mode().iloc[0] if not series.mode().empty else "N/A"
            description = (
                f"Categorical/text column with {unique} unique values. "
                f"Most common value: '{top}'. Missing: {missing}/{total}."
            )
        dictionary[col] = {"type": col_type, "description": description, "missing": missing, "unique": unique}
    return {"dataset_id": dataset_id, "data_dictionary": dictionary}
