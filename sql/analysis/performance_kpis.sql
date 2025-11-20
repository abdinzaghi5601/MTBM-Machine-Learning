-- Key Performance Indicators (KPIs) for Tunneling Operations
-- Advanced analytics queries for business intelligence and reporting

-- =====================================================
-- OPERATIONAL KPIs
-- =====================================================

-- Overall Equipment Effectiveness (OEE) Calculation
WITH operational_time AS (
    SELECT 
        DATE(timestamp) as operation_date,
        COUNT(*) * 0.5 as total_hours,  -- Assuming 30-min intervals
        SUM(CASE WHEN advance_speed > 0 THEN 0.5 ELSE 0 END) as productive_hours,
        AVG(advance_speed) as avg_speed,
        50 as target_speed  -- Target advance speed mm/min
    FROM tunneling_performance
    GROUP BY DATE(timestamp)
)
SELECT 
    operation_date,
    total_hours,
    productive_hours,
    ROUND(productive_hours / total_hours * 100, 2) as availability_percent,
    ROUND(avg_speed / target_speed * 100, 2) as performance_percent,
    ROUND((productive_hours / total_hours) * (avg_speed / target_speed) * 100, 2) as oee_percent
FROM operational_time
ORDER BY operation_date;

-- Advance Rate KPIs
SELECT 
    'Daily Advance Rate' as kpi_name,
    ROUND(AVG(daily_advance), 2) as average_value,
    ROUND(MIN(daily_advance), 2) as minimum_value,
    ROUND(MAX(daily_advance), 2) as maximum_value,
    'meters/day' as unit
FROM (
    SELECT 
        DATE(timestamp) as operation_date,
        MAX(chainage) - MIN(chainage) as daily_advance
    FROM tunneling_performance
    GROUP BY DATE(timestamp)
) daily_progress

UNION ALL

SELECT 
    'Hourly Advance Rate' as kpi_name,
    ROUND(AVG(advance_speed) * 60 / 1000, 2) as average_value,
    ROUND(MIN(advance_speed) * 60 / 1000, 2) as minimum_value,
    ROUND(MAX(advance_speed) * 60 / 1000, 2) as maximum_value,
    'meters/hour' as unit
FROM tunneling_performance;

-- =====================================================
-- QUALITY KPIs
-- =====================================================

-- Deviation Control KPIs
WITH deviation_stats AS (
    SELECT 
        SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2)) as total_deviation
    FROM tunneling_performance
)
SELECT 
    'Average Deviation' as kpi_name,
    ROUND(AVG(total_deviation), 2) as value,
    'mm' as unit,
    CASE 
        WHEN AVG(total_deviation) <= 5 THEN 'Excellent'
        WHEN AVG(total_deviation) <= 10 THEN 'Good'
        WHEN AVG(total_deviation) <= 15 THEN 'Acceptable'
        ELSE 'Poor'
    END as performance_rating
FROM deviation_stats

UNION ALL

SELECT 
    'Maximum Deviation' as kpi_name,
    ROUND(MAX(total_deviation), 2) as value,
    'mm' as unit,
    CASE 
        WHEN MAX(total_deviation) <= 20 THEN 'Within Tolerance'
        ELSE 'Exceeds Tolerance'
    END as performance_rating
FROM deviation_stats

UNION ALL

SELECT 
    'Deviation Standard Deviation' as kpi_name,
    ROUND(STDDEV(total_deviation), 2) as value,
    'mm' as unit,
    CASE 
        WHEN STDDEV(total_deviation) <= 3 THEN 'Consistent'
        WHEN STDDEV(total_deviation) <= 6 THEN 'Moderate Variation'
        ELSE 'High Variation'
    END as performance_rating
FROM deviation_stats;

-- Quality Achievement Rates
SELECT 
    quality_level,
    reading_count,
    ROUND(percentage, 1) as percentage,
    CASE 
        WHEN quality_level = 'Excellent' AND percentage >= 50 THEN 'Target Achieved'
        WHEN quality_level = 'Good' AND percentage >= 30 THEN 'Acceptable'
        WHEN quality_level = 'Poor' AND percentage <= 20 THEN 'Target Achieved'
        ELSE 'Needs Improvement'
    END as target_status
FROM (
    SELECT 
        CASE 
            WHEN SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2)) <= 5 THEN 'Excellent'
            WHEN SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2)) <= 10 THEN 'Good'
            WHEN SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2)) <= 15 THEN 'Acceptable'
            ELSE 'Poor'
        END as quality_level,
        COUNT(*) as reading_count,
        COUNT(*) * 100.0 / (SELECT COUNT(*) FROM tunneling_performance) as percentage
    FROM tunneling_performance
    GROUP BY quality_level
) quality_summary
ORDER BY 
    CASE quality_level 
        WHEN 'Excellent' THEN 1 
        WHEN 'Good' THEN 2 
        WHEN 'Acceptable' THEN 3 
        ELSE 4 
    END;

-- =====================================================
-- EFFICIENCY KPIs
-- =====================================================

-- Energy Efficiency Metrics
SELECT 
    geological_type,
    COUNT(*) as readings,
    ROUND(AVG(total_thrust / advance_speed), 2) as avg_specific_energy,
    ROUND(AVG(advance_speed / revolution_rpm), 2) as avg_cutting_efficiency,
    ROUND(AVG(advance_speed / working_pressure), 2) as avg_pressure_efficiency,
    ROUND(AVG((total_thrust * advance_speed) / 1000), 2) as avg_power_utilization_kw
FROM tunneling_performance
GROUP BY geological_type
ORDER BY avg_cutting_efficiency DESC;

-- Resource Utilization KPIs
SELECT 
    'Grouting Efficiency' as kpi_category,
    ROUND(AVG(grouting_volume), 1) as avg_volume_per_meter,
    ROUND(STDDEV(grouting_volume), 1) as volume_std_dev,
    ROUND(AVG(grouting_pressure), 1) as avg_pressure,
    'L/m' as unit
FROM tunneling_performance

UNION ALL

SELECT 
    'Thrust Utilization' as kpi_category,
    ROUND(AVG(total_thrust), 0) as avg_thrust,
    ROUND(STDDEV(total_thrust), 0) as thrust_std_dev,
    ROUND(MAX(total_thrust), 0) as max_thrust,
    'kN' as unit
FROM tunneling_performance

UNION ALL

SELECT 
    'RPM Utilization' as kpi_category,
    ROUND(AVG(revolution_rpm), 1) as avg_rpm,
    ROUND(STDDEV(revolution_rpm), 1) as rpm_std_dev,
    ROUND(MAX(revolution_rpm), 1) as max_rpm,
    'rpm' as unit
FROM tunneling_performance;

-- =====================================================
-- MAINTENANCE KPIs
-- =====================================================

-- Cutter Wear Analysis
SELECT 
    geological_type,
    COUNT(*) as readings,
    ROUND(AVG(cutter_wear_rate), 4) as avg_wear_rate_mm_hr,
    ROUND(MAX(cutter_wear_rate), 4) as max_wear_rate_mm_hr,
    ROUND(SUM(cutter_wear_rate * 0.5), 2) as estimated_total_wear_mm,
    CASE 
        WHEN AVG(cutter_wear_rate) <= 0.1 THEN 'Low Wear'
        WHEN AVG(cutter_wear_rate) <= 0.5 THEN 'Moderate Wear'
        ELSE 'High Wear'
    END as wear_category
FROM tunneling_performance
GROUP BY geological_type
ORDER BY avg_wear_rate_mm_hr DESC;

-- Predictive Maintenance Indicators
WITH maintenance_indicators AS (
    SELECT 
        timestamp,
        chainage,
        cutter_wear_rate,
        total_thrust,
        advance_speed,
        LAG(cutter_wear_rate, 1) OVER (ORDER BY timestamp) as prev_wear_rate,
        AVG(cutter_wear_rate) OVER (ORDER BY timestamp ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) as wear_trend_10
    FROM tunneling_performance
)
SELECT 
    COUNT(CASE WHEN cutter_wear_rate > wear_trend_10 * 1.5 THEN 1 END) as high_wear_alerts,
    COUNT(CASE WHEN total_thrust > (SELECT AVG(total_thrust) * 1.2 FROM tunneling_performance) THEN 1 END) as high_thrust_alerts,
    COUNT(CASE WHEN advance_speed < (SELECT AVG(advance_speed) * 0.7 FROM tunneling_performance) THEN 1 END) as low_performance_alerts,
    ROUND(
        (COUNT(CASE WHEN cutter_wear_rate > wear_trend_10 * 1.5 THEN 1 END) + 
         COUNT(CASE WHEN total_thrust > (SELECT AVG(total_thrust) * 1.2 FROM tunneling_performance) THEN 1 END) +
         COUNT(CASE WHEN advance_speed < (SELECT AVG(advance_speed) * 0.7 FROM tunneling_performance) THEN 1 END)) * 100.0 / COUNT(*), 2
    ) as alert_percentage
FROM maintenance_indicators;

-- =====================================================
-- COST EFFICIENCY KPIs
-- =====================================================

-- Cost per Meter Analysis (estimated)
WITH cost_estimates AS (
    SELECT 
        geological_type,
        COUNT(*) as readings,
        AVG(advance_speed) as avg_advance_speed,
        AVG(cutter_wear_rate) as avg_wear_rate,
        AVG(grouting_volume) as avg_grout_volume,
        AVG((total_thrust * advance_speed) / 1000) as avg_power_kw,
        -- Estimated costs (example rates)
        AVG(cutter_wear_rate) * 500 as estimated_cutter_cost_per_hour,  -- $500 per mm wear
        AVG(grouting_volume) * 2 as estimated_grout_cost_per_meter,     -- $2 per liter
        AVG((total_thrust * advance_speed) / 1000) * 0.1 as estimated_power_cost_per_hour  -- $0.1 per kWh
    FROM tunneling_performance
    GROUP BY geological_type
)
SELECT 
    geological_type,
    ROUND(avg_advance_speed * 60 / 1000, 2) as meters_per_hour,
    ROUND(estimated_cutter_cost_per_hour / (avg_advance_speed * 60 / 1000), 2) as cutter_cost_per_meter,
    ROUND(estimated_grout_cost_per_meter, 2) as grout_cost_per_meter,
    ROUND(estimated_power_cost_per_hour / (avg_advance_speed * 60 / 1000), 2) as power_cost_per_meter,
    ROUND(
        (estimated_cutter_cost_per_hour + estimated_grout_cost_per_meter + estimated_power_cost_per_hour) / 
        (avg_advance_speed * 60 / 1000), 2
    ) as total_estimated_cost_per_meter
FROM cost_estimates
ORDER BY total_estimated_cost_per_meter;

-- =====================================================
-- BENCHMARKING KPIs
-- =====================================================

-- Performance Benchmarking Against Targets
SELECT 
    'Advance Rate' as performance_metric,
    ROUND(AVG(advance_speed), 2) as actual_value,
    50 as target_value,
    'mm/min' as unit,
    ROUND((AVG(advance_speed) / 50) * 100, 1) as achievement_percentage,
    CASE 
        WHEN AVG(advance_speed) >= 50 THEN 'Target Achieved'
        WHEN AVG(advance_speed) >= 40 THEN 'Close to Target'
        ELSE 'Below Target'
    END as status
FROM tunneling_performance

UNION ALL

SELECT 
    'Deviation Control' as performance_metric,
    ROUND(AVG(SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2))), 2) as actual_value,
    10 as target_value,
    'mm' as unit,
    ROUND((10 / AVG(SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2)))) * 100, 1) as achievement_percentage,
    CASE 
        WHEN AVG(SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2))) <= 10 THEN 'Target Achieved'
        WHEN AVG(SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2))) <= 15 THEN 'Close to Target'
        ELSE 'Below Target'
    END as status
FROM tunneling_performance

UNION ALL

SELECT 
    'Quality Achievement' as performance_metric,
    ROUND(SUM(CASE WHEN SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2)) <= 10 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as actual_value,
    80 as target_value,
    '%' as unit,
    ROUND((SUM(CASE WHEN SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2)) <= 10 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) / 80 * 100, 1) as achievement_percentage,
    CASE 
        WHEN (SUM(CASE WHEN SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2)) <= 10 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) >= 80 THEN 'Target Achieved'
        WHEN (SUM(CASE WHEN SQRT(POWER(horizontal_deviation_machine, 2) + POWER(vertical_deviation_machine, 2)) <= 10 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) >= 60 THEN 'Close to Target'
        ELSE 'Below Target'
    END as status
FROM tunneling_performance;
