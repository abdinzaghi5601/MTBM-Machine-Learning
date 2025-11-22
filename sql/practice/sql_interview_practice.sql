-- SQL Practice Queries for Data Analytics Job Interviews
-- Based on Tunneling Performance Analytics Dataset

-- =====================================================
-- BASIC SQL QUERIES (Entry Level)
-- =====================================================

-- 1. Basic SELECT with filtering
-- Question: Find all records where advance speed is greater than 40 mm/min
SELECT timestamp, chainage, advance_speed, geological_type
FROM tunneling_performance
WHERE advance_speed > 40
ORDER BY advance_speed DESC;

-- 2. Aggregation functions
-- Question: Calculate average, min, max advance speed by geological type
SELECT 
    geological_type,
    COUNT(*) as record_count,
    ROUND(AVG(advance_speed), 2) as avg_advance_speed,
    ROUND(MIN(advance_speed), 2) as min_advance_speed,
    ROUND(MAX(advance_speed), 2) as max_advance_speed
FROM tunneling_performance
GROUP BY geological_type
ORDER BY avg_advance_speed DESC;

-- 3. Date functions
-- Question: Extract performance metrics by day of week
SELECT 
    DAYNAME(timestamp) as day_of_week,
    COUNT(*) as readings,
    ROUND(AVG(advance_speed), 2) as avg_speed,
    ROUND(AVG(SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2))), 2) as avg_deviation
FROM tunneling_performance
GROUP BY DAYNAME(timestamp), DAYOFWEEK(timestamp)
ORDER BY DAYOFWEEK(timestamp);

-- =====================================================
-- INTERMEDIATE SQL QUERIES (Mid Level)
-- =====================================================

-- 4. Subqueries
-- Question: Find records with above-average performance
SELECT 
    timestamp,
    chainage,
    advance_speed,
    geological_type,
    ROUND(advance_speed - (SELECT AVG(advance_speed) FROM tunneling_performance), 2) as speed_vs_average
FROM tunneling_performance
WHERE advance_speed > (SELECT AVG(advance_speed) FROM tunneling_performance)
ORDER BY advance_speed DESC
LIMIT 10;

-- 5. Window functions - Ranking
-- Question: Rank performance within each geological type
SELECT 
    timestamp,
    chainage,
    geological_type,
    advance_speed,
    RANK() OVER (PARTITION BY geological_type ORDER BY advance_speed DESC) as speed_rank,
    ROUND(AVG(advance_speed) OVER (PARTITION BY geological_type), 2) as geo_type_avg
FROM tunneling_performance
ORDER BY geological_type, speed_rank;

-- 6. Moving averages
-- Question: Calculate 5-period moving average of advance speed
SELECT 
    timestamp,
    chainage,
    advance_speed,
    ROUND(AVG(advance_speed) OVER (
        ORDER BY timestamp 
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ), 2) as moving_avg_5,
    ROUND(advance_speed - AVG(advance_speed) OVER (
        ORDER BY timestamp 
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ), 2) as deviation_from_ma
FROM tunneling_performance
ORDER BY timestamp;

-- 7. CASE statements for categorization
-- Question: Categorize performance levels
SELECT 
    timestamp,
    chainage,
    advance_speed,
    CASE 
        WHEN advance_speed >= 50 THEN 'Excellent'
        WHEN advance_speed >= 40 THEN 'Good'
        WHEN advance_speed >= 30 THEN 'Average'
        ELSE 'Below Average'
    END as performance_category,
    SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2)) as total_deviation,
    CASE 
        WHEN SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2)) <= 5 THEN 'Excellent'
        WHEN SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2)) <= 10 THEN 'Good'
        WHEN SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2)) <= 15 THEN 'Acceptable'
        ELSE 'Poor'
    END as quality_category
FROM tunneling_performance
ORDER BY timestamp;

-- =====================================================
-- ADVANCED SQL QUERIES (Senior Level)
-- =====================================================

-- 8. Complex CTEs (Common Table Expressions)
-- Question: Find the best and worst performing sections
WITH section_performance AS (
    SELECT 
        FLOOR(chainage / 50) * 50 as section_start,
        COUNT(*) as readings,
        ROUND(AVG(advance_speed), 2) as avg_speed,
        ROUND(AVG(SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2))), 2) as avg_deviation,
        ROUND(AVG(cutter_wear_rate), 4) as avg_wear_rate
    FROM tunneling_performance
    GROUP BY FLOOR(chainage / 50)
    HAVING COUNT(*) >= 10  -- Only sections with sufficient data
),
performance_ranks AS (
    SELECT 
        *,
        RANK() OVER (ORDER BY avg_speed DESC) as speed_rank,
        RANK() OVER (ORDER BY avg_deviation ASC) as quality_rank,
        (RANK() OVER (ORDER BY avg_speed DESC) + RANK() OVER (ORDER BY avg_deviation ASC)) / 2.0 as combined_rank
    FROM section_performance
)
SELECT 
    section_start,
    section_start + 50 as section_end,
    readings,
    avg_speed,
    avg_deviation,
    avg_wear_rate,
    speed_rank,
    quality_rank,
    combined_rank,
    CASE 
        WHEN combined_rank <= 3 THEN 'Top Performer'
        WHEN combined_rank >= (SELECT COUNT(*) - 2 FROM performance_ranks) THEN 'Needs Improvement'
        ELSE 'Average'
    END as section_category
FROM performance_ranks
ORDER BY combined_rank;

-- 9. Pivot table simulation
-- Question: Create a pivot showing performance metrics by geological type and time period
SELECT 
    geological_type,
    SUM(CASE WHEN HOUR(timestamp) BETWEEN 6 AND 13 THEN 1 ELSE 0 END) as morning_shift_readings,
    ROUND(AVG(CASE WHEN HOUR(timestamp) BETWEEN 6 AND 13 THEN advance_speed END), 2) as morning_avg_speed,
    SUM(CASE WHEN HOUR(timestamp) BETWEEN 14 AND 21 THEN 1 ELSE 0 END) as afternoon_shift_readings,
    ROUND(AVG(CASE WHEN HOUR(timestamp) BETWEEN 14 AND 21 THEN advance_speed END), 2) as afternoon_avg_speed,
    SUM(CASE WHEN HOUR(timestamp) BETWEEN 22 AND 23 OR HOUR(timestamp) BETWEEN 0 AND 5 THEN 1 ELSE 0 END) as night_shift_readings,
    ROUND(AVG(CASE WHEN HOUR(timestamp) BETWEEN 22 AND 23 OR HOUR(timestamp) BETWEEN 0 AND 5 THEN advance_speed END), 2) as night_avg_speed
FROM tunneling_performance
GROUP BY geological_type
ORDER BY geological_type;

-- 10. Correlation analysis
-- Question: Calculate correlation between different performance metrics
WITH correlation_data AS (
    SELECT 
        advance_speed,
        total_thrust,
        working_pressure,
        revolution_rpm,
        SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2)) as total_deviation,
        cutter_wear_rate
    FROM tunneling_performance
),
stats AS (
    SELECT 
        AVG(advance_speed) as avg_speed,
        AVG(total_thrust) as avg_thrust,
        AVG(working_pressure) as avg_pressure,
        AVG(revolution_rpm) as avg_rpm,
        AVG(total_deviation) as avg_deviation,
        STDDEV(advance_speed) as std_speed,
        STDDEV(total_thrust) as std_thrust,
        STDDEV(working_pressure) as std_pressure,
        STDDEV(revolution_rpm) as std_rpm,
        STDDEV(total_deviation) as std_deviation
    FROM correlation_data
)
SELECT 
    'Speed vs Thrust' as metric_pair,
    ROUND(
        AVG((cd.advance_speed - s.avg_speed) * (cd.total_thrust - s.avg_thrust)) / 
        (s.std_speed * s.std_thrust), 3
    ) as correlation_coefficient
FROM correlation_data cd
CROSS JOIN stats s

UNION ALL

SELECT 
    'Speed vs Deviation' as metric_pair,
    ROUND(
        AVG((cd.advance_speed - s.avg_speed) * (cd.total_deviation - s.avg_deviation)) / 
        (s.std_speed * s.std_deviation), 3
    ) as correlation_coefficient
FROM correlation_data cd
CROSS JOIN stats s

UNION ALL

SELECT 
    'Thrust vs Pressure' as metric_pair,
    ROUND(
        AVG((cd.total_thrust - s.avg_thrust) * (cd.working_pressure - s.avg_pressure)) / 
        (s.std_thrust * s.std_pressure), 3
    ) as correlation_coefficient
FROM correlation_data cd
CROSS JOIN stats s;

-- =====================================================
-- DATA QUALITY & VALIDATION QUERIES
-- =====================================================

-- 11. Data quality checks
-- Question: Identify data quality issues
SELECT 
    'Missing Values' as check_type,
    SUM(CASE WHEN advance_speed IS NULL THEN 1 ELSE 0 END) as advance_speed_nulls,
    SUM(CASE WHEN total_thrust IS NULL THEN 1 ELSE 0 END) as thrust_nulls,
    SUM(CASE WHEN geological_type IS NULL THEN 1 ELSE 0 END) as geo_type_nulls,
    COUNT(*) as total_records
FROM tunneling_performance

UNION ALL

SELECT 
    'Outliers (>3 std dev)' as check_type,
    SUM(CASE WHEN ABS(advance_speed - (SELECT AVG(advance_speed) FROM tunneling_performance)) > 
        3 * (SELECT STDDEV(advance_speed) FROM tunneling_performance) THEN 1 ELSE 0 END) as speed_outliers,
    SUM(CASE WHEN ABS(total_thrust - (SELECT AVG(total_thrust) FROM tunneling_performance)) > 
        3 * (SELECT STDDEV(total_thrust) FROM tunneling_performance) THEN 1 ELSE 0 END) as thrust_outliers,
    0 as geo_type_outliers,
    COUNT(*) as total_records
FROM tunneling_performance

UNION ALL

SELECT 
    'Logical Inconsistencies' as check_type,
    SUM(CASE WHEN advance_speed <= 0 THEN 1 ELSE 0 END) as negative_speed,
    SUM(CASE WHEN total_thrust <= 0 THEN 1 ELSE 0 END) as negative_thrust,
    SUM(CASE WHEN working_pressure <= 0 THEN 1 ELSE 0 END) as negative_pressure,
    COUNT(*) as total_records
FROM tunneling_performance;

-- =====================================================
-- BUSINESS INTELLIGENCE QUERIES
-- =====================================================

-- 12. Cohort analysis
-- Question: Analyze performance trends over time
WITH daily_performance AS (
    SELECT 
        DATE(timestamp) as operation_date,
        AVG(advance_speed) as daily_avg_speed,
        AVG(SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2))) as daily_avg_deviation,
        COUNT(*) as daily_readings
    FROM tunneling_performance
    GROUP BY DATE(timestamp)
),
performance_with_lag AS (
    SELECT 
        *,
        LAG(daily_avg_speed, 1) OVER (ORDER BY operation_date) as prev_day_speed,
        LAG(daily_avg_deviation, 1) OVER (ORDER BY operation_date) as prev_day_deviation
    FROM daily_performance
)
SELECT 
    operation_date,
    daily_avg_speed,
    daily_avg_deviation,
    ROUND(daily_avg_speed - prev_day_speed, 2) as speed_change,
    ROUND(daily_avg_deviation - prev_day_deviation, 2) as deviation_change,
    CASE 
        WHEN daily_avg_speed > prev_day_speed AND daily_avg_deviation < prev_day_deviation THEN 'Improving'
        WHEN daily_avg_speed < prev_day_speed AND daily_avg_deviation > prev_day_deviation THEN 'Declining'
        WHEN daily_avg_speed > prev_day_speed THEN 'Speed Up'
        WHEN daily_avg_deviation < prev_day_deviation THEN 'Quality Up'
        ELSE 'Stable'
    END as trend_direction
FROM performance_with_lag
WHERE prev_day_speed IS NOT NULL
ORDER BY operation_date;

-- =====================================================
-- INTERVIEW SCENARIO QUESTIONS
-- =====================================================

-- Scenario 1: "Find the top 10% performers and analyze their characteristics"
WITH performance_percentiles AS (
    SELECT 
        *,
        NTILE(10) OVER (ORDER BY advance_speed DESC) as speed_decile,
        NTILE(10) OVER (ORDER BY SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2)) ASC) as quality_decile
    FROM tunneling_performance
)
SELECT 
    geological_type,
    COUNT(*) as top_performer_count,
    ROUND(AVG(advance_speed), 2) as avg_speed,
    ROUND(AVG(total_thrust), 0) as avg_thrust,
    ROUND(AVG(working_pressure), 0) as avg_pressure,
    ROUND(AVG(revolution_rpm), 1) as avg_rpm
FROM performance_percentiles
WHERE speed_decile = 1 AND quality_decile <= 2  -- Top 10% speed, top 20% quality
GROUP BY geological_type
ORDER BY top_performer_count DESC;

-- Scenario 2: "Identify patterns that predict equipment maintenance needs"
WITH maintenance_indicators AS (
    SELECT 
        timestamp,
        chainage,
        cutter_wear_rate,
        total_thrust,
        advance_speed,
        revolution_rpm,
        AVG(cutter_wear_rate) OVER (ORDER BY timestamp ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) as wear_trend_10,
        AVG(total_thrust) OVER (ORDER BY timestamp ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) as thrust_trend_10,
        STDDEV(advance_speed) OVER (ORDER BY timestamp ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) as speed_volatility_10
    FROM tunneling_performance
)
SELECT 
    timestamp,
    chainage,
    cutter_wear_rate,
    wear_trend_10,
    CASE 
        WHEN cutter_wear_rate > wear_trend_10 * 1.5 THEN 'High Wear Alert'
        WHEN total_thrust > thrust_trend_10 * 1.2 THEN 'High Stress Alert'
        WHEN speed_volatility_10 > (SELECT STDDEV(advance_speed) FROM tunneling_performance) * 1.5 THEN 'Performance Instability'
        ELSE 'Normal'
    END as maintenance_alert,
    ROUND(wear_trend_10 * 100, 2) as projected_wear_next_100_hours  -- Projection
FROM maintenance_indicators
WHERE timestamp >= (SELECT MAX(timestamp) - INTERVAL 7 DAY FROM tunneling_performance)  -- Last week
ORDER BY 
    CASE 
        WHEN cutter_wear_rate > wear_trend_10 * 1.5 THEN 1
        WHEN total_thrust > thrust_trend_10 * 1.2 THEN 2
        ELSE 3
    END,
    timestamp DESC;

-- Scenario 3: "Calculate ROI of performance improvements"
WITH performance_baseline AS (
    SELECT 
        AVG(advance_speed) as baseline_speed,
        AVG(cutter_wear_rate) as baseline_wear_rate,
        AVG(grouting_volume) as baseline_grout_volume
    FROM tunneling_performance
    WHERE timestamp <= (SELECT MIN(timestamp) + INTERVAL 7 DAY FROM tunneling_performance)  -- First week as baseline
),
current_performance AS (
    SELECT 
        AVG(advance_speed) as current_speed,
        AVG(cutter_wear_rate) as current_wear_rate,
        AVG(grouting_volume) as current_grout_volume
    FROM tunneling_performance
    WHERE timestamp >= (SELECT MAX(timestamp) - INTERVAL 7 DAY FROM tunneling_performance)  -- Last week
)
SELECT 
    ROUND(cp.current_speed - pb.baseline_speed, 2) as speed_improvement_mm_min,
    ROUND((cp.current_speed - pb.baseline_speed) / pb.baseline_speed * 100, 1) as speed_improvement_percent,
    ROUND(pb.baseline_wear_rate - cp.current_wear_rate, 4) as wear_reduction_mm_hr,
    ROUND((pb.baseline_wear_rate - cp.current_wear_rate) / pb.baseline_wear_rate * 100, 1) as wear_reduction_percent,
    -- Estimated cost savings (example calculations)
    ROUND((cp.current_speed - pb.baseline_speed) * 60 / 1000 * 24 * 1000, 0) as additional_meters_per_day,  -- Additional progress
    ROUND((pb.baseline_wear_rate - cp.current_wear_rate) * 24 * 500, 0) as daily_cutter_cost_savings_usd  -- Wear cost savings
FROM performance_baseline pb
CROSS JOIN current_performance cp;
