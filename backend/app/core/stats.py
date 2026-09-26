import pandas as pd
import numpy as np
from typing import Dict, Any, List

def generate_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Return summary statistics for each column.
    Numeric columns: count, mean, median, std, min, max, missing.
    Categorical columns: count, unique, top (most frequent), freq, missing.
    """
    summary: Dict[str, Any] = {}
    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        if pd.api.types.is_numeric_dtype(series):
            summary[col] = {
                "type": "numeric",
                "count": int(series.count()),
                "mean": round(series.mean(skipna=True), 4),
                "median": round(series.median(skipna=True), 4),
                "std": round(series.std(skipna=True), 4),
                "min": round(series.min(skipna=True), 4),
                "max": round(series.max(skipna=True), 4),
                "missing": missing,
            }
        else:
            top = None
            freq = 0
            if not series.mode().empty:
                top = series.mode().iloc[0]
                freq = int(series.value_counts().loc[top])
            summary[col] = {
                "type": "categorical",
                "count": int(series.count()),
                "unique": int(series.nunique(dropna=True)),
                "top": top,
                "freq": freq,
                "missing": missing,
            }
    return summary

def generate_histogram(series: pd.Series, bin_count: int = 10) -> List[Dict[str, Any]]:
    """Return histogram data for a numeric series.
    Output list of bins with `x0`, `x1`, `count` keys.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return []
    clean = series.dropna()
    if clean.empty:
        return []
    counts, edges = np.histogram(clean, bins=bin_count)
    bins = []
    for i in range(len(counts)):
        bins.append({"x0": float(edges[i]), "x1": float(edges[i + 1]), "count": int(counts[i])})
    return bins
