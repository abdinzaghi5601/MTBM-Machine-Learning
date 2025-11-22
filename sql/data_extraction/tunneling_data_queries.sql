-- Tunneling Performance Analytics - SQL Queries
-- Data extraction and analysis queries for MTBM operations

-- =====================================================
-- 1. BASIC DATA EXTRACTION QUERIES
-- =====================================================

-- Extract all tunneling performance data with basic metrics
SELECT 
    timestamp,
    chainage,
    geological_type,
    advance_speed,
    revolution_rpm,
    working_pressure,
    total_thrust,
    earth_pressure,
    horizontal_deviation_machine,
    vertical_deviation_machine,
    SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2)) AS total_deviation,
    cutter_wear_rate,
    grouting_volume,
    grouting_pressure
FROM tunneling_performance
ORDER BY timestamp;

-- Extract performance data for specific geological conditions
SELECT 
    geological_type,
    COUNT(*) as record_count,
    AVG(advance_speed) as avg_advance_speed,
    AVG(total_thrust) as avg_thrust,
    AVG(SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2))) as avg_deviation,
    AVG(cutter_wear_rate) as avg_wear_rate
FROM tunneling_performance
GROUP BY geological_type
ORDER BY avg_advance_speed DESC;

-- =====================================================
-- 2. PERFORMANCE ANALYSIS QUERIES
-- =====================================================

-- Daily performance summary
SELECT 
    DATE(timestamp) as operation_date,
    COUNT(*) as readings_count,
    AVG(advance_speed) as avg_advance_speed,
    MIN(advance_speed) as min_advance_speed,
    MAX(advance_speed) as max_advance_speed,
    STDDEV(advance_speed) as std_advance_speed,
    AVG(SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2))) as avg_deviation,
    MAX(chainage) - MIN(chainage) as daily_progress
FROM tunneling_performance
GROUP BY DATE(timestamp)
ORDER BY operation_date;

-- Shift performance comparison
SELECT 
    CASE 
        WHEN HOUR(timestamp) BETWEEN 6 AND 17 THEN 'Day Shift'
        WHEN HOUR(timestamp) BETWEEN 18 AND 23 OR HOUR(timestamp) BETWEEN 0 AND 5 THEN 'Night Shift'
    END as shift_type,
    COUNT(*) as readings,
    AVG(advance_speed) as avg_advance_speed,
    AVG(SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2))) as avg_deviation,
    AVG(cutter_wear_rate) as avg_wear_rate,
    AVG(total_thrust) as avg_thrust
FROM tunneling_performance
GROUP BY shift_type
ORDER BY avg_advance_speed DESC;

-- =====================================================
-- 3. QUALITY CONTROL QUERIES
-- =====================================================

-- Deviation quality analysis
SELECT 
    chainage,
    timestamp,
    geological_type,
    horizontal_deviation_machine,
    vertical_deviation_machine,
    SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2)) as total_deviation,
    CASE 
        WHEN SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2)) <= 5 THEN 'Excellent'
        WHEN SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2)) <= 10 THEN 'Good'
        WHEN SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2)) <= 20 THEN 'Acceptable'
        ELSE 'Poor'
    END as quality_rating
FROM tunneling_performance
ORDER BY total_deviation DESC;

-- Quality distribution summary
SELECT 
    CASE 
        WHEN SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2)) <= 5 THEN 'Excellent'
        WHEN SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2)) <= 10 THEN 'Good'
        WHEN SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2)) <= 20 THEN 'Acceptable'
        ELSE 'Poor'
    END as quality_rating,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM tunneling_performance), 2) as percentage
FROM tunneling_performance
GROUP BY quality_rating
ORDER BY count DESC;

-- =====================================================
-- 4. PREDICTIVE MAINTENANCE QUERIES
-- =====================================================

-- Cutter wear analysis by geological type
SELECT 
    geological_type,
    AVG(cutter_wear_rate) as avg_wear_rate,
    MAX(cutter_wear_rate) as max_wear_rate,
    COUNT(*) as readings,
    SUM(cutter_wear_rate * 0.5) as estimated_total_wear_mm  -- Assuming 0.5 hours per reading
FROM tunneling_performance
GROUP BY geological_type
ORDER BY avg_wear_rate DESC;

-- High wear rate alerts (top 10% wear rates)
WITH wear_percentiles AS (
    SELECT 
        PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY cutter_wear_rate) as p90_wear_rate
    FROM tunneling_performance
)
SELECT 
    tp.timestamp,
    tp.chainage,
    tp.geological_type,
    tp.cutter_wear_rate,
    tp.advance_speed,
    tp.total_thrust,
    'HIGH WEAR ALERT' as alert_type
FROM tunneling_performance tp
CROSS JOIN wear_percentiles wp
WHERE tp.cutter_wear_rate >= wp.p90_wear_rate
ORDER BY tp.cutter_wear_rate DESC;

-- =====================================================
-- 5. OPERATIONAL EFFICIENCY QUERIES
-- =====================================================

-- Efficiency metrics calculation
SELECT 
    timestamp,
    chainage,
    geological_type,
    advance_speed,
    revolution_rpm,
    total_thrust,
    working_pressure,
    -- Calculated efficiency metrics
    total_thrust / advance_speed as specific_energy,
    advance_speed / revolution_rpm as cutting_efficiency,
    advance_speed / working_pressure as pressure_efficiency,
    earth_pressure / advance_speed as ground_resistance
FROM tunneling_performance
ORDER BY timestamp;

-- Best performing sections (top quartile efficiency)
WITH efficiency_metrics AS (
    SELECT 
        *,
        advance_speed / revolution_rpm as cutting_efficiency,
        NTILE(4) OVER (ORDER BY advance_speed / revolution_rpm DESC) as efficiency_quartile
    FROM tunneling_performance
)
SELECT 
    chainage,
    geological_type,
    advance_speed,
    revolution_rpm,
    cutting_efficiency,
    'TOP QUARTILE' as performance_category
FROM efficiency_metrics
WHERE efficiency_quartile = 1
ORDER BY cutting_efficiency DESC;

-- =====================================================
-- 6. TREND ANALYSIS QUERIES
-- =====================================================

-- Moving average performance trends
SELECT 
    chainage,
    timestamp,
    advance_speed,
    AVG(advance_speed) OVER (
        ORDER BY timestamp 
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) as moving_avg_5_periods,
    SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2)) as total_deviation,
    AVG(SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2))) OVER (
        ORDER BY timestamp 
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) as moving_avg_deviation
FROM tunneling_performance
ORDER BY timestamp;

-- Performance correlation analysis
SELECT 
    CORR(advance_speed, total_thrust) as speed_thrust_correlation,
    CORR(advance_speed, working_pressure) as speed_pressure_correlation,
    CORR(advance_speed, revolution_rpm) as speed_rpm_correlation,
    CORR(SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2)), advance_speed) as deviation_speed_correlation
FROM tunneling_performance;

-- =====================================================
-- 7. GEOLOGICAL CONDITION ANALYSIS
-- =====================================================

-- Ground condition impact on performance
SELECT 
    geological_type,
    ucs_strength,
    abrasivity_index,
    COUNT(*) as readings,
    AVG(advance_speed) as avg_advance_speed,
    AVG(total_thrust) as avg_thrust,
    AVG(cutter_wear_rate) as avg_wear_rate,
    AVG(SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2))) as avg_deviation
FROM tunneling_performance
GROUP BY geological_type, 
         ROUND(ucs_strength, -2),  -- Group by hundreds
         ROUND(abrasivity_index, 1)  -- Group by tenths
HAVING COUNT(*) >= 5  -- Only include groups with sufficient data
ORDER BY geological_type, ucs_strength;

-- =====================================================
-- 8. REPORTING QUERIES
-- =====================================================

-- Executive summary report
SELECT 
    'Project Summary' as metric_category,
    COUNT(*) as total_readings,
    ROUND(MAX(chainage), 1) as tunnel_length_m,
    ROUND(AVG(advance_speed), 2) as avg_advance_speed_mm_min,
    ROUND(AVG(SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2))), 2) as avg_deviation_mm,
    ROUND(AVG(cutter_wear_rate), 4) as avg_wear_rate_mm_hr,
    MIN(DATE(timestamp)) as project_start_date,
    MAX(DATE(timestamp)) as project_end_date
FROM tunneling_performance

UNION ALL

SELECT 
    'Quality Performance' as metric_category,
    SUM(CASE WHEN SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2)) <= 10 THEN 1 ELSE 0 END) as good_quality_readings,
    ROUND(SUM(CASE WHEN SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2)) <= 10 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as quality_percentage,
    NULL, NULL, NULL, NULL, NULL
FROM tunneling_performance;

-- Detailed performance by chainage sections
SELECT 
    FLOOR(chainage / 50) * 50 as chainage_section_start,
    FLOOR(chainage / 50) * 50 + 50 as chainage_section_end,
    COUNT(*) as readings,
    AVG(advance_speed) as avg_advance_speed,
    AVG(SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2))) as avg_deviation,
    AVG(cutter_wear_rate) as avg_wear_rate,
    STRING_AGG(DISTINCT geological_type, ', ') as geological_types
FROM tunneling_performance
GROUP BY FLOOR(chainage / 50)
ORDER BY chainage_section_start;
