# Power BI Dashboard Structure - Tunneling Performance Analytics

## Dashboard Overview

This document outlines the structure and design for Power BI dashboards to visualize tunneling performance analytics data.

## 📊 Dashboard 1: Executive Summary

### Key Metrics Cards
- **Total Tunnel Length**: 499.5m
- **Project Duration**: 21 days
- **Average Advance Rate**: 33.3 mm/min
- **Overall Quality Score**: 13.3% (Good/Excellent)
- **Equipment Utilization**: 87.3%

### Visualizations

#### 1. Daily Progress Chart (Line Chart)
- **X-axis**: Date
- **Y-axis**: Daily advance (meters)
- **Secondary Y-axis**: Cumulative progress
- **Filters**: Date range, geological type

#### 2. Quality Distribution (Donut Chart)
- **Values**: Count of readings by quality level
- **Legend**: Excellent (0.8%), Good (4.2%), Acceptable (0%), Poor (95%)
- **Colors**: Green, Yellow, Orange, Red

#### 3. Performance by Geological Type (Clustered Column Chart)
- **X-axis**: Geological type (Hard Rock, Soft Clay, Dense Sand, Mixed Ground)
- **Y-axis**: Average advance speed (mm/min)
- **Secondary series**: Average deviation (mm)

#### 4. Trend Analysis (Line Chart)
- **X-axis**: Chainage (tunnel position)
- **Y-axis**: Total deviation (mm)
- **Color by**: Geological type
- **Trend line**: Moving average

## 📈 Dashboard 2: Operational Performance

### Key Performance Indicators
- **Current Advance Speed**: 33.3 mm/min
- **Target Achievement**: 66.6% of target (50 mm/min)
- **Deviation Control**: 8.9mm average
- **Cutter Wear Rate**: 0.33 mm/hr average

### Visualizations

#### 1. Real-time Performance Gauge
- **Metric**: Current advance speed
- **Range**: 0-80 mm/min
- **Target line**: 50 mm/min
- **Color coding**: Red (<30), Yellow (30-45), Green (>45)

#### 2. Machine Parameters Heatmap
- **Rows**: Time periods (hourly)
- **Columns**: Parameters (Thrust, RPM, Pressure, Speed)
- **Color scale**: Performance efficiency
- **Tooltips**: Actual values and targets

#### 3. Shift Performance Comparison (Bar Chart)
- **X-axis**: Shift type (Day/Night)
- **Y-axis**: Average performance metrics
- **Series**: Advance speed, deviation, efficiency
- **Filters**: Date range, crew

#### 4. Equipment Utilization Timeline
- **X-axis**: Time
- **Y-axis**: Equipment status
- **Color coding**: Active (Green), Maintenance (Yellow), Downtime (Red)
- **Drill-through**: Detailed downtime analysis

## 🔧 Dashboard 3: Maintenance & Quality

### Maintenance Indicators
- **Next Maintenance Due**: 45 hours
- **Cutter Life Remaining**: 67%
- **High Wear Alerts**: 3 active
- **Maintenance Cost/Meter**: $125

### Visualizations

#### 1. Cutter Wear Prediction (Line Chart with Forecast)
- **X-axis**: Operating hours
- **Y-axis**: Cumulative wear (mm)
- **Historical data**: Actual wear
- **Forecast line**: Predicted wear
- **Alert threshold**: Replacement point

#### 2. Quality Control Matrix
- **Rows**: Tunnel sections (50m intervals)
- **Columns**: Quality metrics (Deviation, Alignment, Grouting)
- **Color scale**: Quality score (Red-Yellow-Green)
- **Interactive**: Click for detailed section analysis

#### 3. Maintenance Schedule Gantt Chart
- **Y-axis**: Equipment components
- **X-axis**: Timeline
- **Bars**: Scheduled maintenance windows
- **Markers**: Actual maintenance events
- **Color coding**: Preventive vs. corrective

#### 4. Cost Analysis (Waterfall Chart)
- **Categories**: Labor, Materials, Equipment, Downtime
- **Values**: Cost per meter by category
- **Comparison**: Budget vs. actual
- **Drill-down**: Detailed cost breakdown

## 🌍 Dashboard 4: Geological Analysis

### Ground Condition Metrics
- **Current Ground Type**: Hard Rock
- **UCS Strength**: 2,500 kPa
- **Abrasivity Index**: 0.8
- **Groundwater Level**: 15m

### Visualizations

#### 1. Geological Profile (Area Chart)
- **X-axis**: Chainage (tunnel position)
- **Y-axis**: Ground type layers
- **Colors**: Different geological types
- **Overlay**: Performance metrics
- **Interactive**: Hover for ground properties

#### 2. Performance vs. Ground Conditions (Scatter Plot)
- **X-axis**: UCS strength
- **Y-axis**: Advance speed
- **Bubble size**: Cutter wear rate
- **Color**: Geological type
- **Trend lines**: Performance correlation

#### 3. Ground Condition Impact Matrix
- **Rows**: Ground types
- **Columns**: Performance metrics
- **Values**: Average performance by ground type
- **Heat map**: Color-coded performance levels

#### 4. Predictive Ground Model (Map Visual)
- **Base**: Tunnel alignment
- **Layers**: Predicted vs. actual ground conditions
- **Markers**: Survey points and machine readings
- **Color coding**: Prediction accuracy

## 📱 Mobile Dashboard: Field Operations

### Key Mobile Metrics
- **Current Position**: Ch 125+50
- **Live Advance Speed**: 42 mm/min
- **Deviation Status**: 8.3mm (Good)
- **Next Alert**: Maintenance in 2 hours

### Mobile Visualizations

#### 1. Performance Speedometer
- **Single large gauge**: Current advance speed
- **Target indicator**: Performance vs. target
- **Color zones**: Performance categories

#### 2. Alert Summary Cards
- **Active alerts**: Count and priority
- **Recent events**: Last 4 hours
- **Quick actions**: Acknowledge, escalate

#### 3. Trend Mini-Charts
- **Last 24 hours**: Key performance trends
- **Compact format**: Sparklines for mobile
- **Touch interaction**: Tap for details

## 🔄 Data Refresh & Filters

### Refresh Schedule
- **Real-time data**: Every 30 seconds (operational dashboards)
- **Historical data**: Every 15 minutes
- **Reports**: Daily at 6 AM
- **Maintenance data**: Every hour

### Global Filters
- **Date Range**: Custom date picker
- **Tunnel Section**: Chainage range selector
- **Geological Type**: Multi-select dropdown
- **Shift**: Day/Night toggle
- **Equipment Status**: Active/Maintenance filter

### Drill-through Pages
- **Section Detail**: Detailed analysis for specific tunnel sections
- **Equipment Detail**: Component-level performance and maintenance
- **Crew Performance**: Individual and team performance metrics
- **Cost Analysis**: Detailed cost breakdown and trends

## 📊 Data Sources & Connections

### Primary Data Sources
1. **Tunneling Performance Database**
   - Connection: SQL Server
   - Refresh: Real-time via DirectQuery
   - Tables: tunneling_performance, geological_profile

2. **Maintenance System**
   - Connection: REST API
   - Refresh: Hourly
   - Data: Maintenance schedules, work orders, costs

3. **Quality Control System**
   - Connection: Excel/CSV import
   - Refresh: Daily
   - Data: Survey results, deviation measurements

### Calculated Measures (DAX)

```dax
-- Total Deviation
Total Deviation = SQRT(POWER([Horizontal Deviation], 2) + POWER([Vertical Deviation], 2))

-- Alignment Quality Score
Alignment Quality = 1 / (1 + [Total Deviation])

-- Performance Achievement
Performance Achievement = [Actual Advance Speed] / [Target Advance Speed]

-- Quality Rating
Quality Rating = 
SWITCH(
    TRUE(),
    [Total Deviation] <= 5, "Excellent",
    [Total Deviation] <= 10, "Good",
    [Total Deviation] <= 15, "Acceptable",
    "Poor"
)

-- Moving Average (5 periods)
Advance Speed MA5 = 
AVERAGEX(
    DATESINPERIOD(
        'Date'[Date],
        LASTDATE('Date'[Date]),
        -5,
        DAY
    ),
    [Average Advance Speed]
)

-- Cutter Wear Prediction
Predicted Wear = 
[Current Wear] + ([Average Wear Rate] * [Hours Remaining])

-- Cost per Meter
Cost per Meter = 
DIVIDE(
    [Total Operating Cost],
    [Total Meters Completed],
    0
)
```

## 🎨 Design Guidelines

### Color Scheme
- **Primary**: Blue (#1f77b4) - Performance metrics
- **Secondary**: Orange (#ff7f0e) - Alerts and warnings
- **Success**: Green (#2ca02c) - Good performance
- **Warning**: Yellow (#ffbb78) - Caution
- **Danger**: Red (#d62728) - Poor performance/alerts

### Typography
- **Headers**: Segoe UI Bold, 16pt
- **Subheaders**: Segoe UI Semibold, 12pt
- **Body text**: Segoe UI Regular, 10pt
- **Metrics**: Segoe UI Bold, 24pt (large numbers)

### Layout Principles
- **Grid system**: 12-column responsive grid
- **White space**: Adequate spacing between elements
- **Hierarchy**: Clear visual hierarchy with size and color
- **Consistency**: Standardized chart types and formatting

## 📋 Implementation Checklist

### Phase 1: Core Dashboards
- [ ] Set up data connections
- [ ] Create executive summary dashboard
- [ ] Implement real-time operational dashboard
- [ ] Design maintenance and quality dashboard

### Phase 2: Advanced Features
- [ ] Add geological analysis dashboard
- [ ] Implement mobile-responsive design
- [ ] Create drill-through pages
- [ ] Set up automated refresh schedules

### Phase 3: Enhancement
- [ ] Add predictive analytics visuals
- [ ] Implement alert system integration
- [ ] Create custom visuals if needed
- [ ] User acceptance testing and feedback

### Phase 4: Deployment
- [ ] Publish to Power BI Service
- [ ] Set up row-level security
- [ ] Configure sharing and permissions
- [ ] Create user documentation and training

## 🔐 Security & Permissions

### User Roles
- **Executives**: View-only access to summary dashboards
- **Operations Managers**: Full access to operational dashboards
- **Field Engineers**: Mobile dashboard access with real-time data
- **Maintenance Team**: Maintenance dashboard with edit capabilities
- **Data Analysts**: Full access to all dashboards and underlying data

### Data Security
- **Row-level security**: Filter data by project/section access
- **Column-level security**: Restrict sensitive cost/performance data
- **Refresh credentials**: Secure service account for data connections
- **Audit logging**: Track dashboard usage and data access
