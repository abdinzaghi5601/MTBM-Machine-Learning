# MTBM ML Framework - Quick Start Guide

## What Changed?

Your project now has an organized structure! All outputs are automatically saved to designated folders:

### New Folder Structure

```
ML for Tunneling/
├── data/
│   ├── raw/              ← Put original CSV files here
│   └── processed/        ← Generated/processed data saves here
│
├── outputs/
│   ├── plots/            ← All PNG visualizations save here
│   ├── reports/          ← Analysis reports save here
│   ├── models/           ← ML models save here
│   └── logs/             ← Execution logs save here
│
├── scripts/              ← Utility scripts
├── docs/                 ← Documentation (guides you created)
└── MTBM-Machine-Learning/  ← Main code repository
```

## How to Run Your Analysis

### Step 1: Open Terminal/Command Prompt

```bash
cd "C:\Users\abdul\Desktop\ML for Tunneling"
```

### Step 2: Run the Main Script

```bash
cd MTBM-Machine-Learning
python mtbm_comprehensive_plotting.py
```

### Step 3: Find Your Outputs

After running, you'll find:

**Plots** (in `outputs/plots/`):
- `mtbm_time_series_overview.png` - All parameters over time
- `mtbm_deviation_analysis.png` - Tunnel alignment quality
- `mtbm_performance_dashboard.png` - Operational efficiency
- `mtbm_correlation_matrix.png` - Parameter relationships

**Data** (in `data/processed/`):
- `mtbm_comprehensive_data.csv` - Complete processed dataset

## Understanding Your Results

### 📊 Read the Documentation

Three comprehensive guides are now available in the `docs/` folder:

1. **PLOT_INTERPRETATION_GUIDE.md** (Most Important!)
   - How to read each plot
   - What indicates good performance ✅
   - What indicates problems ⚠️
   - Decision-making workflows
   - **Start here to understand your results!**

2. **CODE_STRUCTURE_GUIDE.md**
   - How the code works
   - How to customize it
   - Adding new parameters
   - Troubleshooting

3. **Main README.md** (Project root)
   - Project overview
   - Installation instructions
   - Feature list
   - Quality standards reference

### 🎯 Quick Interpretation Checklist

After generating plots, check:

**✅ Good Outcomes**:
- Deviation plot: Points mostly in green circle (±25mm)
- Time series: Smooth trends, no erratic spikes
- Performance: Drilling efficiency stable around 0.18-0.25
- Correlation: Strong correlation between pressure and speed

**⚠️ Warning Signs**:
- Deviation plot: Points outside orange circle (>50mm)
- Time series: Sudden jumps or declining trends
- Performance: Efficiency declining over time
- Correlation: Unexpected patterns or weak expected correlations

**🛑 Critical Issues** (Stop and Investigate):
- Deviation plot: Points outside red circle (>75mm)
- Time series: Multiple parameters showing erratic behavior
- Performance: Temperature >35°C, speed <15mm/min
- Correlation: Multiple unexpected strong correlations

## Using Your Own Data

Instead of synthetic data, use your actual MTBM measurements:

### Option 1: Replace in Code

```python
# In mtbm_comprehensive_plotting.py, main() function
# Replace line ~591:
# df = plotter.generate_synthetic_mtbm_data(n_samples=1000)

# With:
df = pd.read_csv('../data/raw/YOUR_DATA_FILE.csv')
```

### Option 2: Quick Script

Create a new file `analyze_my_data.py`:

```python
import pandas as pd
from mtbm_comprehensive_plotting import MTBMComprehensivePlotter

# Load your data
df = pd.read_csv('../data/raw/your_actual_data.csv')

# Create plotter
plotter = MTBMComprehensivePlotter()

# Generate all plots
plotter.plot_time_series_overview(df)
plotter.plot_deviation_analysis(df)
plotter.plot_performance_dashboard(df)
plotter.plot_correlation_matrix(df)
plotter.generate_comprehensive_report(df)
```

### Required Column Names

Your CSV must have these columns (or rename them):
- `timestamp` - DateTime of measurement
- `tunnel_length_m` - Tunnel length in meters
- `hor_deviation_machine_mm` - Horizontal deviation
- `vert_deviation_machine_mm` - Vertical deviation
- `earth_pressure_01_bar` - Earth pressure
- `working_pressure_bar` - Working pressure
- `advance_speed_mm_min` - Advance speed
- `revolution_rpm` - Cutter wheel RPM
- And other parameters as needed

## Common Tasks

### Task 1: Generate Fresh Plots

```bash
cd MTBM-Machine-Learning
python mtbm_comprehensive_plotting.py
```

Check `outputs/plots/` for PNG files.

### Task 2: Analyze Existing CSV

```bash
# Put CSV in data/raw/
# Modify script to load it (see "Using Your Own Data" above)
python mtbm_comprehensive_plotting.py
```

### Task 3: Compare Before/After

```bash
# Run analysis on old data
python mtbm_comprehensive_plotting.py
# Files saved with default names

# Manually rename output files
cd ../outputs/plots
rename mtbm_deviation_analysis.png mtbm_deviation_analysis_OLD.png

# Run analysis on new data
python mtbm_comprehensive_plotting.py
# New files generated

# Now compare *_OLD.png with current *.png
```

### Task 4: Export for Report

All plots are already high-resolution (300 DPI) PNG files ready for:
- PowerPoint presentations
- PDF reports
- Technical documentation
- Email/sharing

Just copy from `outputs/plots/` and paste into your document.

## Troubleshooting

### "ModuleNotFoundError: No module named 'matplotlib'"

Install required packages:

```bash
pip install matplotlib seaborn pandas numpy
```

Or if using virtual environment:

```bash
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

pip install matplotlib seaborn pandas numpy
```

### "Permission denied" when saving files

**Cause**: CSV file is open in Excel or another program

**Solution**: Close all Excel files and run again

### Plots not appearing in outputs/plots/

**Check**:
1. Script finished without errors?
2. Check console output for save paths
3. Directory permissions (can you create files there?)

### Data looks wrong in plots

**Common issues**:
- Decimal points in wrong place (earth pressure should be 8-26, not 80-260)
- Missing timestamps (can't plot time series without time data)
- Column names don't match expected format

**Solution**: Review data preprocessing, check the earth pressure fix we did earlier.

## Next Steps

1. **Run the script** with synthetic data to see how it works
2. **Read PLOT_INTERPRETATION_GUIDE.md** to understand results
3. **Load your actual data** and generate real analysis
4. **Review the plots** using the interpretation guide
5. **Make decisions** based on what the data shows

## Key Quality Standards Reference

| Metric | Excellent | Good | Acceptable | Poor/Critical |
|--------|-----------|------|------------|---------------|
| Total Deviation | <25mm | 26-50mm | 51-75mm | >75mm |
| Earth Pressure | 12-20 bar | 8-26 bar | 5-30 bar | <5 or >30 |
| Advance Speed | 30-40 mm/min | 20-45 mm/min | 15-50 mm/min | <15 or >50 |
| Drilling Efficiency | 0.20-0.25 | 0.15-0.30 | 0.10-0.35 | <0.10 or >0.35 |
| Temperature | 20-25°C | 15-30°C | 10-35°C | <10 or >35 |

## Getting Help

1. **Plot interpretation**: Read `docs/PLOT_INTERPRETATION_GUIDE.md`
2. **Code questions**: Read `docs/CODE_STRUCTURE_GUIDE.md`
3. **Project overview**: Read `README.md`
4. **Specific issues**: Check troubleshooting section above

## Remember

- **Trends matter more than single points** - Look for patterns over time
- **Compare with standards** - Use the quality thresholds
- **Document findings** - Note unusual patterns for engineering review
- **Regular monitoring** - Generate plots daily/weekly to catch issues early

---

**You're all set!** The framework is ready to use. Start by running the script and reviewing the generated plots with the interpretation guide.

**Created**: November 2024
**Version**: 1.0
