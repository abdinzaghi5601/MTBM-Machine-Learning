# MTBM Machine Learning Framework

A comprehensive machine learning and data analysis framework for Microtunneling Boring Machine (MTBM) operations. This project analyzes 23+ key operational parameters to optimize tunneling performance, predict maintenance needs, and ensure precise tunnel alignment.

## Project Structure

```
ML for Tunneling/
├── data/
│   ├── raw/              # Original, unprocessed data files
│   └── processed/        # Cleaned and processed datasets
│
├── outputs/
│   ├── plots/            # All generated visualizations
│   ├── reports/          # Analysis reports and summaries
│   ├── models/           # Trained ML models
│   └── logs/             # Execution logs
│
├── scripts/              # Utility and helper scripts
│
├── docs/                 # Documentation files
│
├── MTBM-Machine-Learning/    # Main ML framework repository
│
└── README.md             # This file
```

## Key Features

### 1. Comprehensive Parameter Monitoring (23+ Parameters)

#### Survey Position Parameters
- Tunnel length progression
- Horizontal deviation (machine & drill head)
- Vertical deviation (machine & drill head)
- Total deviation tracking

#### Survey Orientation Parameters
- Yaw (steering angle)
- Pitch (vertical angle)
- Reel angle
- Temperature monitoring (ELS/MWD)
- Survey mode tracking (ELS, ELS-HWL, GNS)

#### Steering Control Parameters
- 4 hydraulic cylinder stroke positions
- Total steering force
- Real-time steering adjustments

#### Operational Parameters
- Advance speed (mm/min)
- Interjack force
- Active interjack tracking

#### Cutter Wheel Parameters
- Working pressure
- Revolution speed (RPM)
- Earth pressure in excavation chamber

### 2. Data Analysis & Visualization

The framework generates four main types of visualizations:

1. **Time Series Overview** (`mtbm_time_series_overview.png`)
2. **Deviation Analysis** (`mtbm_deviation_analysis.png`)
3. **Performance Dashboard** (`mtbm_performance_dashboard.png`)
4. **Correlation Matrix** (`mtbm_correlation_matrix.png`)

## Getting Started

### Prerequisites

```bash
python >= 3.8
pandas
numpy
matplotlib
seaborn
scikit-learn
```

### Installation

1. Navigate to the project directory:
```bash
cd "ML for Tunneling"
```

2. Activate virtual environment (if available):
```bash
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate     # On Windows
```

3. Install required packages:
```bash
pip install -r ml_requirements.txt
```

### Running the Analysis

To generate comprehensive analysis and visualizations:

```bash
cd MTBM-Machine-Learning
python mtbm_comprehensive_plotting.py
```

This will:
- Generate synthetic MTBM operational data (or use your actual data)
- Create all visualizations in `outputs/plots/`
- Save processed data in `data/processed/`
- Display a comprehensive operational report

## Understanding the Outputs

### Data Files

**Location**: `data/processed/`

- `mtbm_comprehensive_data.csv`: Complete processed dataset with all 23+ parameters

### Visualization Files

**Location**: `outputs/plots/`

1. **mtbm_time_series_overview.png**
   - Shows trends for all parameters over time
   - Helps identify operational patterns and anomalies
   - Organized by parameter groups (position, orientation, steering, operational, cutter wheel)

2. **mtbm_deviation_analysis.png**
   - Tunnel deviation patterns (horizontal vs vertical)
   - Deviation tolerance circles (±25mm, ±50mm, ±75mm)
   - Steering cylinder response analysis
   - Correlation between steering force and deviation

3. **mtbm_performance_dashboard.png**
   - Speed vs pressure performance
   - Drilling efficiency trends
   - Pressure balance analysis
   - Interjack force distribution
   - Temperature monitoring
   - Survey mode usage distribution

4. **mtbm_correlation_matrix.png**
   - Shows relationships between all parameters
   - Identifies key performance drivers
   - Helps understand parameter interdependencies

## Quality Standards & Thresholds

### Tunnel Deviation Quality Levels

| Category | Total Deviation | Color Code | Interpretation |
|----------|----------------|------------|----------------|
| Excellent | ≤ 25mm | Green | Optimal alignment |
| Good | 26-50mm | Yellow | Within acceptable tolerance |
| Acceptable | 51-75mm | Orange | Requires attention |
| Poor | > 75mm | Red | Immediate correction needed |

### Operational Parameters - Normal Ranges

| Parameter | Normal Range | Units | Critical Thresholds |
|-----------|-------------|-------|---------------------|
| Earth Pressure | 8-26 | bar | <5 or >30 |
| Working Pressure | 120-200 | bar | <100 or >220 |
| Advance Speed | 15-45 | mm/min | <10 or >50 |
| Revolution Speed | 6-12 | RPM | <5 or >15 |
| Temperature (ELS) | 15-30 | °C | <10 or >35 |
| Interjack Force | 800-1500 | kN | <600 or >1700 |
| Total Steering Force | 500-750 | kN | <400 or >900 |

### Performance Metrics

- **Drilling Efficiency**: Advance speed / Working pressure (typical: 0.15-0.30 mm/min/bar)
- **Power Efficiency**: Advance speed / Revolution speed (typical: 2.5-5.0 mm/min/RPM)

## Interpreting the Visualizations

### Good Operational Outcomes

✅ **Deviation Analysis**
- Points clustered near center (0,0) on deviation scatter plot
- Most deviations within green tolerance circle (±25mm)
- Smooth deviation trends without sudden jumps
- Balanced steering cylinder movements

✅ **Performance Dashboard**
- Advance speed consistently in 20-40 mm/min range
- Drilling efficiency trend relatively stable
- Earth pressure closely matching working pressure (1:1 ratio)
- Temperature staying within 20-30°C range

✅ **Time Series Overview**
- Smooth parameter trends without erratic behavior
- Cylinder strokes moving in coordinated patterns
- Gradual tunnel length progression
- Minimal deviation fluctuations

### Alarming Instances (Require Attention)

⚠️ **Critical Warning Signs**

1. **Deviation Issues**
   - Total deviation exceeding 50mm consistently
   - Points outside orange tolerance circle (75mm)
   - Sudden large jumps in deviation
   - Asymmetric steering cylinder positions

2. **Performance Problems**
   - Advance speed dropping below 15 mm/min
   - Drilling efficiency declining sharply
   - Large gap between earth pressure and working pressure
   - Temperature exceeding 30°C

3. **Mechanical Concerns**
   - Cylinder strokes at extreme positions (<20mm or >80mm)
   - Total steering force exceeding 750 kN
   - Interjack force showing irregular spikes
   - Revolution speed fluctuating erratically

4. **Correlation Matrix Red Flags**
   - Unexpected strong correlations (|r| > 0.8) between unrelated parameters
   - Loss of expected correlations (e.g., pressure vs speed)
   - Indicates potential sensor issues or unusual ground conditions

## Data Quality & Troubleshooting

### Common Issues

1. **File Permission Errors**
   - Close all CSV files before running scripts
   - Ensure write permissions for output directories

2. **Missing Visualizations**
   - Check that output directories exist
   - Verify matplotlib backend is properly configured

3. **Data Scale Issues**
   - Earth pressure should be -10 to +30 bar (not 100-300)
   - Check decimal point positions in raw data

## AVN Protocol Support

This framework supports multiple AVN protocols:
- AVN 800
- AVN 1200
- AVN 2400
- AVN 3000

Protocol-specific measurement files are included in the project root.

## Contributing

When adding new analysis scripts:
1. Save outputs to appropriate `outputs/` subdirectories
2. Process data files go to `data/processed/`
3. Raw data files go to `data/raw/`
4. Document new parameters and thresholds

## License

This project is for tunneling operation analysis and optimization.

## Contact

For questions about interpreting results or operational recommendations, consult with tunnel engineering specialists.

---

**Last Updated**: November 2024
**Framework Version**: 1.0
**Data Parameters**: 23+ operational metrics
