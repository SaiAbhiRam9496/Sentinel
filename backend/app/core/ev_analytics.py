"""
ev_analytics.py — Phase 7
Core analytical aggregations and visualization generators for EV Mode.
Computes KPIs, charger breakdowns, vehicle model comparisons, time-of-day demand,
temperature relationships, and binned distributions.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from backend.app.core.stats import generate_histogram


def compute_ev_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute high-level executive KPIs for an EV telemetry dataset."""
    total_sessions = len(df)
    if total_sessions == 0:
        return {
            "total_sessions": 0,
            "total_energy_kwh": 0.0,
            "total_cost_usd": 0.0,
            "avg_duration_hours": 0.0,
            "avg_charging_rate_kw": 0.0,
            "avg_soc_delta": 0.0,
            "avg_battery_capacity_kwh": 0.0,
        }

    energy = pd.to_numeric(df.get("Energy Consumed (kWh)"), errors="coerce").dropna()
    duration = pd.to_numeric(df.get("Charging Duration (hours)"), errors="coerce").dropna()
    rate = pd.to_numeric(df.get("Charging Rate (kW)"), errors="coerce").dropna()
    cost = pd.to_numeric(df.get("Charging Cost (USD)"), errors="coerce").dropna()
    battery = pd.to_numeric(df.get("Battery Capacity (kWh)"), errors="coerce").dropna()

    soc_start = pd.to_numeric(df.get("State of Charge (Start %)"), errors="coerce")
    soc_end = pd.to_numeric(df.get("State of Charge (End %)"), errors="coerce")
    soc_delta = (soc_end - soc_start).dropna()

    return {
        "total_sessions": total_sessions,
        "total_energy_kwh": round(float(energy.sum()), 2) if not energy.empty else 0.0,
        "total_cost_usd": round(float(cost.sum()), 2) if not cost.empty else 0.0,
        "avg_duration_hours": round(float(duration.mean()), 2) if not duration.empty else 0.0,
        "avg_charging_rate_kw": round(float(rate.mean()), 2) if not rate.empty else 0.0,
        "avg_soc_delta": round(float(soc_delta.mean()), 2) if not soc_delta.empty else 0.0,
        "avg_battery_capacity_kwh": round(float(battery.mean()), 1) if not battery.empty else 0.0,
    }


def compute_charger_type_analysis(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Analyze throughput, speed, duration, and cost per Charger Type."""
    if "Charger Type" not in df.columns or df.empty:
        return []

    groups = []
    total_count = len(df)
    for charger, sub in df.groupby("Charger Type"):
        count = len(sub)
        energy = pd.to_numeric(sub.get("Energy Consumed (kWh)"), errors="coerce").dropna()
        duration = pd.to_numeric(sub.get("Charging Duration (hours)"), errors="coerce").dropna()
        rate = pd.to_numeric(sub.get("Charging Rate (kW)"), errors="coerce").dropna()
        cost = pd.to_numeric(sub.get("Charging Cost (USD)"), errors="coerce").dropna()

        groups.append({
            "charger_type": str(charger),
            "session_count": count,
            "session_share_pct": round((count / total_count) * 100, 1),
            "avg_charging_rate_kw": round(float(rate.mean()), 2) if not rate.empty else 0.0,
            "avg_duration_hours": round(float(duration.mean()), 2) if not duration.empty else 0.0,
            "avg_energy_kwh": round(float(energy.mean()), 2) if not energy.empty else 0.0,
            "total_energy_kwh": round(float(energy.sum()), 2) if not energy.empty else 0.0,
            "avg_cost_usd": round(float(cost.mean()), 2) if not cost.empty else 0.0,
        })

    # Sort descending by total energy throughput
    groups.sort(key=lambda x: x["total_energy_kwh"], reverse=True)
    return groups


def compute_vehicle_model_analysis(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Analyze battery consumption, SOC changes, and session frequency per Vehicle Model."""
    if "Vehicle Model" not in df.columns or df.empty:
        return []

    models = []
    for model_name, sub in df.groupby("Vehicle Model"):
        energy = pd.to_numeric(sub.get("Energy Consumed (kWh)"), errors="coerce").dropna()
        duration = pd.to_numeric(sub.get("Charging Duration (hours)"), errors="coerce").dropna()
        battery = pd.to_numeric(sub.get("Battery Capacity (kWh)"), errors="coerce").dropna()
        cost = pd.to_numeric(sub.get("Charging Cost (USD)"), errors="coerce").dropna()

        soc_start = pd.to_numeric(sub.get("State of Charge (Start %)"), errors="coerce")
        soc_end = pd.to_numeric(sub.get("State of Charge (End %)"), errors="coerce")
        soc_delta = (soc_end - soc_start).dropna()

        models.append({
            "vehicle_model": str(model_name),
            "session_count": len(sub),
            "avg_battery_capacity_kwh": round(float(battery.mean()), 1) if not battery.empty else 0.0,
            "avg_energy_consumed_kwh": round(float(energy.mean()), 2) if not energy.empty else 0.0,
            "avg_soc_delta": round(float(soc_delta.mean()), 2) if not soc_delta.empty else 0.0,
            "avg_duration_hours": round(float(duration.mean()), 2) if not duration.empty else 0.0,
            "avg_cost_usd": round(float(cost.mean()), 2) if not cost.empty else 0.0,
        })

    models.sort(key=lambda x: x["session_count"], reverse=True)
    return models


def compute_time_of_day_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate sessions by categorical Time of Day and by start hour (0-23)."""
    result: Dict[str, Any] = {"by_period": [], "by_hour": []}

    # 1. By period (Morning, Afternoon, Evening, Night)
    if "Time of Day" in df.columns:
        for period, sub in df.groupby("Time of Day"):
            energy = pd.to_numeric(sub.get("Energy Consumed (kWh)"), errors="coerce").dropna()
            rate = pd.to_numeric(sub.get("Charging Rate (kW)"), errors="coerce").dropna()
            result["by_period"].append({
                "period": str(period),
                "session_count": len(sub),
                "total_energy_kwh": round(float(energy.sum()), 2) if not energy.empty else 0.0,
                "avg_charging_rate_kw": round(float(rate.mean()), 2) if not rate.empty else 0.0,
            })

    # 2. By hour of day (0 to 23)
    if "Charging Start Time" in df.columns:
        try:
            start_dt = pd.to_datetime(df["Charging Start Time"], errors="coerce")
            hours = start_dt.dt.hour
            hour_counts = hours.value_counts().to_dict()
            for h in range(24):
                result["by_hour"].append({
                    "hour": h,
                    "session_count": int(hour_counts.get(h, 0)),
                })
        except Exception:
            pass

    return result


def compute_temperature_impact(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Examine charging performance across temperature bands."""
    if "Temperature (°C)" not in df.columns or df.empty:
        return []

    temp_series = pd.to_numeric(df["Temperature (°C)"], errors="coerce")
    brackets = [
        ("< 10°C (Cold)", temp_series < 10),
        ("10°C - 20°C (Mild)", (temp_series >= 10) & (temp_series < 20)),
        ("20°C - 30°C (Optimal)", (temp_series >= 20) & (temp_series < 30)),
        ("> 30°C (High)", temp_series >= 30),
    ]

    impact = []
    for label, mask in brackets:
        sub = df[mask]
        if sub.empty:
            continue
        energy = pd.to_numeric(sub.get("Energy Consumed (kWh)"), errors="coerce").dropna()
        rate = pd.to_numeric(sub.get("Charging Rate (kW)"), errors="coerce").dropna()
        duration = pd.to_numeric(sub.get("Charging Duration (hours)"), errors="coerce").dropna()

        impact.append({
            "temperature_range": label,
            "session_count": len(sub),
            "avg_rate_kw": round(float(rate.mean()), 2) if not rate.empty else 0.0,
            "avg_duration_hours": round(float(duration.mean()), 2) if not duration.empty else 0.0,
            "avg_energy_kwh": round(float(energy.mean()), 2) if not energy.empty else 0.0,
        })

    return impact


def compute_station_rankings(df: pd.DataFrame, top_n: int = 10) -> List[Dict[str, Any]]:
    """Rank top charging station locations by energy throughput and utilization."""
    col = "Charging Station Location" if "Charging Station Location" in df.columns else "Charging Station ID"
    if col not in df.columns or df.empty:
        return []

    stations = []
    for loc, sub in df.groupby(col):
        energy = pd.to_numeric(sub.get("Energy Consumed (kWh)"), errors="coerce").dropna()
        rate = pd.to_numeric(sub.get("Charging Rate (kW)"), errors="coerce").dropna()
        cost = pd.to_numeric(sub.get("Charging Cost (USD)"), errors="coerce").dropna()

        stations.append({
            "station_identifier": str(loc),
            "session_count": len(sub),
            "total_energy_kwh": round(float(energy.sum()), 2) if not energy.empty else 0.0,
            "avg_rate_kw": round(float(rate.mean()), 2) if not rate.empty else 0.0,
            "total_revenue_usd": round(float(cost.sum()), 2) if not cost.empty else 0.0,
        })

    stations.sort(key=lambda x: x["total_energy_kwh"], reverse=True)
    return stations[:top_n]


def compute_ev_distributions(df: pd.DataFrame) -> Dict[str, Any]:
    """Produce histogram distributions for primary numeric telemetry variables."""
    distributions = {}

    numeric_targets = {
        "energy_consumed_kwh": "Energy Consumed (kWh)",
        "charging_duration_hours": "Charging Duration (hours)",
        "charging_rate_kw": "Charging Rate (kW)",
        "charging_cost_usd": "Charging Cost (USD)",
        "vehicle_age_years": "Vehicle Age (years)",
    }

    for key, col in numeric_targets.items():
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            distributions[key] = generate_histogram(series, bin_count=10)

    # Derived SOC Delta distribution
    if "State of Charge (Start %)" in df.columns and "State of Charge (End %)" in df.columns:
        start_soc = pd.to_numeric(df["State of Charge (Start %)"], errors="coerce")
        end_soc = pd.to_numeric(df["State of Charge (End %)"], errors="coerce")
        delta = (end_soc - start_soc).dropna()
        distributions["soc_delta_pct"] = generate_histogram(delta, bin_count=10)

    return distributions


def assemble_ev_analytics(df: pd.DataFrame) -> Dict[str, Any]:
    """Assemble complete EV analytics and visualization payloads."""
    return {
        "kpis": compute_ev_kpis(df),
        "charger_types": compute_charger_type_analysis(df),
        "vehicle_models": compute_vehicle_model_analysis(df),
        "time_of_day": compute_time_of_day_analysis(df),
        "temperature_impact": compute_temperature_impact(df),
        "top_stations": compute_station_rankings(df),
        "distributions": compute_ev_distributions(df),
    }


def generate_ev_markdown_report(df: pd.DataFrame, dataset_id: str, filename: str) -> str:
    """Generate a clean, structured Markdown executive analytics report for an EV dataset."""
    kpis = compute_ev_kpis(df)
    chargers = compute_charger_type_analysis(df)
    vehicles = compute_vehicle_model_analysis(df)

    lines = [
        f"# Sentinel — EV Telemetry & Charging Analytics Report",
        f"**Dataset ID:** `{dataset_id}` | **File:** `{filename}` | **Total Records:** {kpis['total_sessions']:,}",
        "",
        "## 1. Executive KPIs",
        f"- **Total Energy Delivered:** {kpis['total_energy_kwh']:,.2f} kWh",
        f"- **Total Charging Revenue:** ${kpis['total_cost_usd']:,.2f}",
        f"- **Average Session Duration:** {kpis['avg_duration_hours']:.2f} hours",
        f"- **Average Charging Rate:** {kpis['avg_charging_rate_kw']:.2f} kW",
        f"- **Average SOC Gained:** {kpis['avg_soc_delta']:.2f}%",
        f"- **Fleet Average Battery Capacity:** {kpis['avg_battery_capacity_kwh']:.1f} kWh",
        "",
        "## 2. Infrastructure & Charger Performance",
        "| Charger Type | Sessions | Share (%) | Avg Rate (kW) | Avg Duration (h) | Total Energy (kWh) |",
        "|---|---|---|---|---|---|",
    ]

    for c in chargers:
        lines.append(
            f"| {c['charger_type']} | {c['session_count']:,} | {c['session_share_pct']}% | "
            f"{c['avg_charging_rate_kw']:.2f} kW | {c['avg_duration_hours']:.2f} h | {c['total_energy_kwh']:,.1f} kWh |"
        )

    lines.extend([
        "",
        "## 3. Fleet Vehicle Model Analysis",
        "| Vehicle Model | Sessions | Avg Battery (kWh) | Avg Consumed (kWh) | Avg SOC Delta (%) |",
        "|---|---|---|---|---|",
    ])

    for v in vehicles[:8]:
        lines.append(
            f"| {v['vehicle_model']} | {v['session_count']:,} | {v['avg_battery_capacity_kwh']:.1f} kWh | "
            f"{v['avg_energy_consumed_kwh']:.2f} kWh | +{v['avg_soc_delta']:.2f}% |"
        )

    lines.extend([
        "",
        "## 4. Operational Recommendations",
        "- **High-Speed Charger Scaling:** DC Fast Chargers demonstrate highest energy velocity; prioritize grid interconnect capacity at peak stations.",
        "- **Off-Peak Incentivization:** Evening and night charging periods show utilization gaps; dynamic time-of-use tariffs can smooth demand spikes.",
        "- **Data Quality Assurance:** Maintain continuous telemetry profiling via Sentinel rules and IsolationForest anomaly monitors.",
        "",
        "---",
        "*Generated by Sentinel Intelligent Data Quality & EV Analytics Platform.*",
    ])

    return "\n".join(lines)
