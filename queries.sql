-- ==============================================================================
-- Sentinel — Core Analytics & Data Engineering Raw SQL Queries
-- ==============================================================================
-- This file provides visible, optimized raw SQL implementations for all core
-- platform analytics, audit logging, and data quality metrics.
-- Per handoff specification Section 9 & Phase 11 hardening.
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- 1. DATASET METADATA & RECENT UPLOADS AUDIT
-- ------------------------------------------------------------------------------
-- Retrieves the most recent dataset uploads along with processing status,
-- row counts, column counts, and detected ingestion mode.

SELECT 
    d.id AS dataset_id,
    d.filename,
    d.detected_mode,
    d.mode_reason,
    d.row_count,
    d.col_count,
    d.created_at,
    COUNT(c.id) AS total_cleaning_actions
FROM datasets d
LEFT JOIN cleaning_logs c ON d.id = c.dataset_id
GROUP BY d.id, d.filename, d.detected_mode, d.mode_reason, d.row_count, d.col_count, d.created_at
ORDER BY d.created_at DESC
LIMIT 20;


-- ------------------------------------------------------------------------------
-- 2. DATA CLEANING & TRANSPARENCY AUDIT TRAIL
-- ------------------------------------------------------------------------------
-- Aggregates data cleaning actions (dropped rows, imputed values, anomaly training)
-- per dataset to substantiate the "what we changed and why" transparency screen.

SELECT 
    dataset_id,
    action,
    COUNT(*) AS action_occurrences,
    MIN(created_at) AS first_logged_at,
    MAX(created_at) AS last_logged_at
FROM cleaning_logs
WHERE dataset_id = :dataset_id
GROUP BY dataset_id, action
ORDER BY action_occurrences DESC;


-- ------------------------------------------------------------------------------
-- 3. EV CHARGER TYPE PERFORMANCE & UTILIZATION
-- ------------------------------------------------------------------------------
-- Calculates energy throughput, average session duration, and effective rate
-- grouped by charger classification (Level 1, Level 2, DC Fast Charger).
-- Used by the EV Analytics dashboard.

WITH session_metrics AS (
    SELECT
        "Charger Type" AS charger_type,
        CAST("Energy Consumed (kWh)" AS FLOAT) AS energy_kwh,
        CAST("Charging Duration (hours)" AS FLOAT) AS duration_hours,
        CAST("Charging Rate (kW)" AS FLOAT) AS charging_rate_kw,
        CAST("Charging Cost (USD)" AS FLOAT) AS cost_usd
    FROM ev_charging_sessions
    WHERE "Charging Duration (hours)" > 0 AND "Energy Consumed (kWh)" > 0
)
SELECT
    charger_type,
    COUNT(*) AS session_count,
    ROUND(AVG(duration_hours)::numeric, 2) AS avg_duration_hours,
    ROUND(AVG(energy_kwh)::numeric, 2) AS avg_energy_kwh,
    ROUND(AVG(charging_rate_kw)::numeric, 2) AS avg_charging_rate_kw,
    ROUND(SUM(energy_kwh)::numeric, 2) AS total_energy_delivered_kwh,
    ROUND(SUM(cost_usd)::numeric, 2) AS total_revenue_usd,
    ROUND((SUM(cost_usd) / NULLIF(SUM(energy_kwh), 0))::numeric, 3) AS effective_cost_per_kwh
FROM session_metrics
GROUP BY charger_type
ORDER BY total_energy_delivered_kwh DESC;


-- ------------------------------------------------------------------------------
-- 4. VEHICLE MODEL CHARGING EFFICIENCY & CONSUMPTION
-- ------------------------------------------------------------------------------
-- Computes average battery utilization, SOC gained per session, and
-- energy delivered relative to stated battery capacity per vehicle model.

SELECT
    "Vehicle Model" AS vehicle_model,
    COUNT(*) AS total_sessions,
    ROUND(AVG(CAST("Battery Capacity (kWh)" AS FLOAT))::numeric, 1) AS avg_battery_capacity_kwh,
    ROUND(AVG(CAST("Energy Consumed (kWh)" AS FLOAT))::numeric, 2) AS avg_energy_consumed_kwh,
    ROUND(AVG(CAST("State of Charge (End %)" AS FLOAT) - CAST("State of Charge (Start %)" AS FLOAT))::numeric, 2) AS avg_soc_delta_pct,
    ROUND(AVG(
        (CAST("Energy Consumed (kWh)" AS FLOAT) / NULLIF(CAST("Battery Capacity (kWh)" AS FLOAT), 0)) * 100
    )::numeric, 2) AS avg_battery_throughput_pct
FROM ev_charging_sessions
WHERE "Vehicle Model" IS NOT NULL
GROUP BY "Vehicle Model"
ORDER BY total_sessions DESC;


-- ------------------------------------------------------------------------------
-- 5. TIME-OF-DAY DEMAND DISTRIBUTION & PEAK HOURS
-- ------------------------------------------------------------------------------
-- Analyzes session volume and average charging rate by start hour of the day
-- to identify grid stress periods and peak utilization windows.

SELECT
    EXTRACT(HOUR FROM CAST("Charging Start Time" AS TIMESTAMP)) AS start_hour_of_day,
    COUNT(*) AS session_count,
    ROUND(SUM(CAST("Energy Consumed (kWh)" AS FLOAT))::numeric, 2) AS total_energy_kwh,
    ROUND(AVG(CAST("Charging Rate (kW)" AS FLOAT))::numeric, 2) AS avg_charging_rate_kw,
    ROUND(AVG(CAST("Charging Cost (USD)" AS FLOAT))::numeric, 2) AS avg_cost_usd
FROM ev_charging_sessions
WHERE "Charging Start Time" IS NOT NULL
GROUP BY EXTRACT(HOUR FROM CAST("Charging Start Time" AS TIMESTAMP))
ORDER BY start_hour_of_day ASC;


-- ------------------------------------------------------------------------------
-- 6. TOP CHARGING HUBS BY THROUGHPUT & SESSION INTENSITY
-- ------------------------------------------------------------------------------
-- Aggregates metrics by station location to spotlight top-demand nodes
-- and geographic charging infrastructure distribution.

SELECT
    "Charging Station Location" AS station_location,
    COUNT(DISTINCT "Charging Station ID") AS distinct_stations,
    COUNT(*) AS total_charging_sessions,
    ROUND(SUM(CAST("Energy Consumed (kWh)" AS FLOAT))::numeric, 2) AS total_energy_kwh,
    ROUND(AVG(CAST("Charging Duration (hours)" AS FLOAT))::numeric, 2) AS avg_duration_hours,
    ROUND(SUM(CAST("Charging Cost (USD)" AS FLOAT))::numeric, 2) AS total_revenue_usd
FROM ev_charging_sessions
WHERE "Charging Station Location" IS NOT NULL
GROUP BY "Charging Station Location"
ORDER BY total_energy_kwh DESC
LIMIT 10;


-- ------------------------------------------------------------------------------
-- 7. STATISTICAL OUTLIER DETECTION (RULE-BASED SANITY CHECKS)
-- ------------------------------------------------------------------------------
-- Identifies sessions with anomalous duration or energy using 95th percentile
-- thresholds as a rule-based pre-filter distinct from ML IsolationForest.

WITH percentiles AS (
    SELECT
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY CAST("Charging Duration (hours)" AS FLOAT)) AS p95_duration,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY CAST("Energy Consumed (kWh)" AS FLOAT)) AS p95_energy
    FROM ev_charging_sessions
)
SELECT
    s."User ID",
    s."Vehicle Model",
    s."Charger Type",
    CAST(s."Charging Duration (hours)" AS FLOAT) AS duration_hours,
    CAST(s."Energy Consumed (kWh)" AS FLOAT) AS energy_kwh,
    CAST(s."Charging Rate (kW)" AS FLOAT) AS rate_kw,
    CASE
        WHEN CAST(s."Charging Duration (hours)" AS FLOAT) > p.p95_duration THEN 'Extreme Duration (>P95)'
        WHEN CAST(s."Energy Consumed (kWh)" AS FLOAT) > p.p95_energy THEN 'Extreme Energy (>P95)'
        ELSE 'Normal Range'
    END AS statistical_flag
FROM ev_charging_sessions s
CROSS JOIN percentiles p
WHERE CAST(s."Charging Duration (hours)" AS FLOAT) > p.p95_duration
   OR CAST(s."Energy Consumed (kWh)" AS FLOAT) > p.p95_energy
LIMIT 50;
