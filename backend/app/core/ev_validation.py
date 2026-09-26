"""
ev_validation.py — Phase 5
EV-specific validity and consistency rules (Section 4 of handoff).
Each rule check returns a list of violations with row index, column, rule, and details.
These are SEPARATE from ML anomaly detection output — never mix the two.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from backend.app.db.models import CleaningLog


# --------------------------------------------------------------------------- #
# Helper                                                                        #
# --------------------------------------------------------------------------- #

def _log(db: Session, dataset_id: str, action: str, details: Dict[str, Any]) -> None:
    db.add(CleaningLog(dataset_id=dataset_id, action=action, details=details))
    db.commit()


# --------------------------------------------------------------------------- #
# EV schema validation rules                                                   #
# --------------------------------------------------------------------------- #

def check_soc_range(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """State of Charge (start and end) must be in [0, 100]."""
    violations = []
    for col in ["State of Charge (Start %)", "State of Charge (End %)"]:
        if col not in df.columns:
            continue
        ser = pd.to_numeric(df[col], errors="coerce")
        bad = df[(ser < 0) | (ser > 100)]
        for idx in bad.index:
            violations.append({
                "row_index": int(idx),
                "column": col,
                "rule": "soc_range",
                "value": df.at[idx, col],
                "reason": f"{col} must be between 0 and 100; got {df.at[idx, col]}",
            })
    return violations


def check_time_order(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Charging End Time must be after Charging Start Time."""
    violations = []
    if "Charging Start Time" not in df.columns or "Charging End Time" not in df.columns:
        return violations
    start = pd.to_datetime(df["Charging Start Time"], errors="coerce")
    end = pd.to_datetime(df["Charging End Time"], errors="coerce")
    bad = df[end <= start]
    for idx in bad.index:
        violations.append({
            "row_index": int(idx),
            "column": "Charging End Time",
            "rule": "time_order",
            "value": str(df.at[idx, "Charging End Time"]),
            "reason": "Charging End Time must be after Charging Start Time",
        })
    return violations


def check_energy_vs_battery(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Energy Consumed (kWh) should not exceed Battery Capacity (kWh)."""
    violations = []
    if "Energy Consumed (kWh)" not in df.columns or "Battery Capacity (kWh)" not in df.columns:
        return violations
    energy = pd.to_numeric(df["Energy Consumed (kWh)"], errors="coerce")
    capacity = pd.to_numeric(df["Battery Capacity (kWh)"], errors="coerce")
    bad = df[energy > capacity]
    for idx in bad.index:
        violations.append({
            "row_index": int(idx),
            "column": "Energy Consumed (kWh)",
            "rule": "energy_vs_battery",
            "value": df.at[idx, "Energy Consumed (kWh)"],
            "reason": (
                f"Energy Consumed ({df.at[idx, 'Energy Consumed (kWh)']} kWh) exceeds "
                f"Battery Capacity ({df.at[idx, 'Battery Capacity (kWh)']} kWh)"
            ),
        })
    return violations


def check_positive_numeric(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Duration, rate, cost, distance, battery capacity must be non-negative."""
    violations = []
    positive_cols = [
        "Charging Duration (hours)",
        "Charging Rate (kW)",
        "Charging Cost (USD)",
        "Distance Driven (since last charge) (km)",
        "Battery Capacity (kWh)",
    ]
    for col in positive_cols:
        if col not in df.columns:
            continue
        ser = pd.to_numeric(df[col], errors="coerce")
        bad = df[ser < 0]
        for idx in bad.index:
            violations.append({
                "row_index": int(idx),
                "column": col,
                "rule": "positive_numeric",
                "value": df.at[idx, col],
                "reason": f"{col} must be non-negative; got {df.at[idx, col]}",
            })
    return violations


def check_soc_start_lt_end(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """After a charging session, SOC end should be >= SOC start (charging only adds charge)."""
    violations = []
    if "State of Charge (Start %)" not in df.columns or "State of Charge (End %)" not in df.columns:
        return violations
    soc_start = pd.to_numeric(df["State of Charge (Start %)"], errors="coerce")
    soc_end = pd.to_numeric(df["State of Charge (End %)"], errors="coerce")
    bad = df[soc_end < soc_start]
    for idx in bad.index:
        violations.append({
            "row_index": int(idx),
            "column": "State of Charge (End %)",
            "rule": "soc_end_lt_start",
            "value": df.at[idx, "State of Charge (End %)"],
            "reason": (
                f"SOC End ({df.at[idx, 'State of Charge (End %)']}%) < "
                f"SOC Start ({df.at[idx, 'State of Charge (Start %)']}%) — impossible for a charging session"
            ),
        })
    return violations


def run_ev_validation(df: pd.DataFrame, dataset_id: str, db: Session) -> Dict[str, Any]:
    """Run all EV-specific rule-based validation checks. Log results and return summary."""
    violations: List[Dict[str, Any]] = []
    violations.extend(check_soc_range(df))
    violations.extend(check_time_order(df))
    violations.extend(check_energy_vs_battery(df))
    violations.extend(check_positive_numeric(df))
    violations.extend(check_soc_start_lt_end(df))

    _log(db, dataset_id, "ev_validation", {
        "total_violations": len(violations),
        "rules_checked": ["soc_range", "time_order", "energy_vs_battery", "positive_numeric", "soc_end_lt_start"],
    })

    return {
        "violation_count": len(violations),
        "violations": violations,
    }
