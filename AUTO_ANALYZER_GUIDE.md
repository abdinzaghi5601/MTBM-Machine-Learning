# Automatic MTBM Protocol Analyzer

## The Simplest Way to Analyze Your Tunnel Data

Just provide your CSV file and pipe diameter - the system does everything else automatically!

---

## Quick Start (30 Seconds)

```bash
cd MTBM-Machine-Learning

# That's it! Just provide your CSV and pipe diameter:
python auto_protocol_analyzer.py --data your_tunnel_data.csv --diameter 800
```

The system automatically:
- ✅ Detects which AVN protocol (800/1200/2400/3000) from your CSV structure
- ✅ Validates the detection with confidence scoring
- ✅ Runs complete standard analysis with protocol-specific settings
- ✅ Performs ML anomaly detection (5 algorithms)
- ✅ Checks pipe bore tolerance compliance
- ✅ Generates integrated summary report

**No manual protocol selection needed!**

---

## What Happens Automatically

### 1. Protocol Auto-Detection

The system analyzes your CSV column structure and determines the protocol:

**AVN 800** - Basic protocol
- Looks for: tunnel_length, position, deviations
- Minimum: 6 core parameters

**AVN 1200** - With orientation
- Looks for: yaw, pitch angles
- Minimum: 8 parameters

**AVN 2400** - Advanced measurement
- Looks for: drill_head, hydraulic pressures
- Minimum: 10 parameters

**AVN 3000** - Complete monitoring
- Looks for: survey_mode, ring_build data
- Minimum: 12 parameters

**Confidence Score**: Reports how certain the detection is (0-100%)

### 2. Complete Analysis Suite

Once protocol is detected, runs:

**Standard Analysis**
- Time series plots for all parameters
- Deviation analysis with protocol-specific thresholds
- Performance dashboard
- Correlation analysis

**ML Anomaly Detection**
- 5 algorithms working together
- Horizontal/vertical deviation patterns
- Severity classification (Low/Medium/High)
- Visual marking of anomalies

**Tolerance Compliance**
- Industry-standard checks based on pipe diameter
- Quality ratings (Excellent → Poor)
- Compliance visualization
- Exceedance percentages

**Integrated Summary**
- Combined report across all analyses
- Critical findings identification
- Statistics summary
- Complete results CSV

---

## Usage Examples

### Basic Analysis
```bash
python auto_protocol_analyzer.py --data tunnel_data.csv --diameter 800
```

### High Sensitivity Anomaly Detection
```bash
python auto_protocol_analyzer.py --data tunnel_data.csv --diameter 1200 --sensitivity high
```

### Skip Certain Analyses
```bash
# Skip anomaly detection (faster)
python auto_protocol_analyzer.py --data tunnel_data.csv --diameter 600 --skip-anomaly

# Skip tolerance check
python auto_protocol_analyzer.py --data tunnel_data.csv --diameter 800 --skip-tolerance
```

### Generate Sample Data for Testing
```bash
python auto_protocol_analyzer.py --generate-sample --diameter 800
```

---

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--data` | Path to your CSV file | Required* |
| `--diameter` | Pipe bore diameter in mm | Required |
| `--sensitivity` | Anomaly detection sensitivity (low/medium/high) | medium |
| `--skip-anomaly` | Skip anomaly detection | False |
| `--skip-tolerance` | Skip tolerance compliance check | False |
| `--generate-sample` | Generate test data instead of loading CSV | False |
| `--samples` | Number of samples to generate (with --generate-sample) | 1000 |

*Either `--data` or `--generate-sample` is required

---

## Understanding the Output

### Console Output

```
================================================================================
AUTOMATIC MTBM PROTOCOL ANALYZER
================================================================================

Step 1: Protocol Auto-Detection
--------------------------------------------------------------------------------
Protocol Detected: AVN2400
Confidence: 95.0%

Detection Details:
  Required parameters matched: 10/10
  Optional parameters matched: 8/12

Applying AVN2400 Configuration...
  ✓ Deviation thresholds: Excellent <25mm, Good <50mm, Poor >75mm
  ✓ Pipe diameter: 800mm
  ✓ Tolerances: ±25mm vertical, ±40mm horizontal

[... continues with analysis results ...]
```

### Output Files

All results saved to: `outputs/[PROTOCOL]/auto_analysis/`

```
outputs/
└── AVN2400/                    ← Detected protocol
    ├── plots/                  ← Standard analysis plots
    │   ├── time_series_*.png
    │   ├── deviation_*.png
    │   └── dashboard_*.png
    └── auto_analysis/          ← Complete analysis results
        ├── anomaly_detection/
        │   ├── anomalies_*.png
        │   ├── anomaly_report_*.txt
        │   └── anomaly_results_*.csv
        ├── tolerance_compliance/
        │   ├── compliance_*.png
        │   ├── compliance_report_*.txt
        │   └── compliance_results_*.csv
        ├── integrated_summary_*.txt    ← START HERE!
        └── complete_results_*.csv      ← Full data with all flags
```

### Integrated Summary Report

**Read this first!** Contains:

1. **Detection Information**
   - Which protocol was detected
   - Confidence score
   - What parameters were found

2. **Deviation Statistics**
   - Mean, std, max for horizontal, vertical, total deviation
   - Compared against protocol thresholds

3. **Anomaly Detection Summary** (if not skipped)
   - Number and percentage of anomalies
   - Breakdown by severity (Low/Medium/High)

4. **Tolerance Compliance Summary** (if not skipped)
   - Compliance rate
   - Quality distribution
   - Out-of-tolerance count

5. **Critical Findings**
   - Automatic identification of issues:
     - High anomaly rate (>20%)
     - Low compliance rate (<80%)
     - Excessive high-severity anomalies
   - Or: "✅ No critical issues detected"

### Complete Results CSV

Your original data **PLUS** new analysis columns:

**From Protocol Analysis:**
- Standard metrics and calculated fields

**From Anomaly Detection:**
- `anomaly_ensemble`: 1 = anomaly, 0 = normal
- `anomaly_score`: 0-1 (confidence)
- `anomaly_severity`: Low/Medium/High
- Individual model predictions

**From Tolerance Compliance:**
- `hor_tolerance_mm`: Applicable horizontal tolerance
- `vert_tolerance_mm`: Applicable vertical tolerance
- `both_within_tolerance`: TRUE/FALSE
- `quality_rating`: Excellent/Good/Acceptable/Marginal/Poor
- `hor_exceedance_pct`: % of tolerance used
- `vert_exceedance_pct`: % of tolerance used

---

## Pipe Diameter Reference

Common diameters and their tolerances:

| Diameter (mm) | Vertical Tolerance | Horizontal Tolerance |
|---------------|-------------------|---------------------|
| 600 | ± 20 mm | ± 25 mm |
| 800 | ± 25 mm | ± 40 mm |
| 1000 | ± 30 mm | ± 100 mm |
| 1200 | ± 30 mm | ± 100 mm |
| 1400 | ± 50 mm | ± 200 mm |
| 1600 | ± 50 mm | ± 200 mm |

The system automatically applies the correct tolerances based on your diameter!

---

## Sensitivity Levels Explained

**Low Sensitivity** (~5% flagged as anomalies)
- Best for: Normal operations, established tunnels
- Reduces false alarms
- Only flags clear outliers

**Medium Sensitivity** (~10% flagged) - **DEFAULT**
- Best for: Standard monitoring
- Balanced approach
- Recommended for most cases

**High Sensitivity** (~15% flagged)
- Best for: Investigation mode, critical sections
- Catches subtle issues
- More false positives, but safer

---

## Troubleshooting

### "Could not detect protocol with confidence"

**Cause**: CSV columns don't match any known protocol well enough

**Solutions**:
1. Check your CSV has the standard column names
2. Ensure required columns are present (see detection output)
3. Manually specify protocol using `analyze_protocol.py` instead

### "Too many anomalies detected"

**Solutions**:
```bash
# Try lower sensitivity
python auto_protocol_analyzer.py --data file.csv --diameter 800 --sensitivity low
```

### "No anomalies found"

**Solutions**:
```bash
# Try higher sensitivity
python auto_protocol_analyzer.py --data file.csv --diameter 800 --sensitivity high
```

### "TensorFlow warning"

**Not an error!** The autoencoder is optional. 4 other ML algorithms still work.

**To enable it**: `pip install tensorflow`

---

## Python API Usage

Want to integrate into your own scripts?

```python
from auto_protocol_analyzer import AutoCSVAnalyzer

# Create analyzer
analyzer = AutoCSVAnalyzer()

# Run complete analysis
results = analyzer.load_and_analyze(
    csv_path='tunnel_data.csv',
    diameter_mm=800,
    sensitivity='medium',
    skip_anomaly=False,
    skip_tolerance=False
)

# Access detected protocol
print(f"Detected: {results['protocol']}")
print(f"Confidence: {results['confidence']:.1f}%")

# Access analyzed dataframe
df = results['dataframe']

# Get critical cases
critical = df[
    (df['anomaly_severity'] == 'High') &
    (~df['both_within_tolerance'])
]
print(f"Critical issues: {len(critical)}")
```

---

## Comparison with Manual Methods

### Old Way (Manual)
```bash
# 1. Look at CSV columns manually
# 2. Decide which protocol
# 3. Run standard analysis
python analyze_protocol.py --protocol AVN2400 --data file.csv

# 4. Run anomaly detection separately
python analyze_with_anomalies.py --protocol AVN2400 --data file.csv

# 5. Run tolerance check with pip_bore_tolerances.py
# 6. Manually combine results
```

### New Way (Automatic)
```bash
# Everything in one command!
python auto_protocol_analyzer.py --data file.csv --diameter 800
```

---

## Decision Guide

### What Result Quality Looks Like

**Excellent Project** ✅
- <5% anomalies detected
- >95% tolerance compliance
- Mostly "Excellent" or "Good" quality ratings
- No critical findings
- **Action**: Proceed with confidence

**Good Project** ⚠️
- 5-10% anomalies
- 85-95% tolerance compliance
- Mostly "Good" or "Acceptable" ratings
- Few critical findings
- **Action**: Review anomalies, monitor trends

**Concerning Project** 🛑
- >15% anomalies
- <80% tolerance compliance
- "Marginal" or "Poor" ratings
- Multiple critical findings
- **Action**:
  - Review integrated summary
  - Investigate high-severity anomalies
  - Check tolerance violations
  - Consider operational adjustments

---

## Best Practices

1. **Start Simple**
   ```bash
   python auto_protocol_analyzer.py --data file.csv --diameter 800
   ```

2. **Review Integrated Summary First**
   - Go to `outputs/[PROTOCOL]/auto_analysis/integrated_summary_*.txt`
   - Read detection info and critical findings

3. **Check Visualizations**
   - Anomaly plots show where issues occur
   - Tolerance plots show quality distribution

4. **Investigate High-Severity Cases**
   - Open `complete_results_*.csv`
   - Filter for `anomaly_severity == 'High'`
   - Filter for `both_within_tolerance == FALSE`

5. **Adjust Sensitivity as Needed**
   - Too many alerts? Use `--sensitivity low`
   - Missing issues? Use `--sensitivity high`

6. **Document Your Findings**
   - Save the integrated summary reports
   - Note any patterns in anomalies
   - Track compliance rates over time

---

## Related Documentation

- **COMPLETE_ANALYSIS_GUIDE.md** - Detailed guide for full system
- **ANOMALY_DETECTION_QUICKSTART.md** - Deep dive into ML algorithms
- **MULTI_PROTOCOL_QUICKSTART.md** - Manual protocol selection
- **docs/PLOT_INTERPRETATION_GUIDE.md** - Understanding visualizations
- **docs/ANOMALY_DETECTION_GUIDE.md** - ML algorithm details

---

## Example Workflow

### Morning Analysis Routine

```bash
# 1. Run analysis on yesterday's data
python auto_protocol_analyzer.py --data yesterday.csv --diameter 800

# 2. Check integrated summary
cat outputs/AVN*/auto_analysis/integrated_summary_*.txt | grep "CRITICAL FINDINGS" -A 10

# 3. If critical findings found:
#    - Open anomaly plots
#    - Review complete_results CSV
#    - Investigate specific chainage ranges

# 4. If all clear:
#    - File the reports
#    - Move to next dataset
```

### Investigation Mode

```bash
# Use high sensitivity for detailed investigation
python auto_protocol_analyzer.py \
    --data problem_section.csv \
    --diameter 800 \
    --sensitivity high

# Review all anomalies and tolerance violations
```

### Quick Check

```bash
# Skip detailed anomaly detection for speed
python auto_protocol_analyzer.py \
    --data file.csv \
    --diameter 800 \
    --skip-anomaly
```

---

## You're Ready!

**Most common command:**
```bash
python auto_protocol_analyzer.py --data your_data.csv --diameter 800
```

**Then read:**
1. `integrated_summary_*.txt` - Overview and critical findings
2. Anomaly visualizations - See detected issues
3. Tolerance plots - Quality assessment
4. `complete_results_*.csv` - Full data for detailed analysis

**Everything you need for comprehensive MTBM tunnel analysis - automatically!** 🚀

---

**Created**: November 2024
**Version**: 1.0
**Features**: Auto Protocol Detection + Standard Analysis + ML Anomaly Detection + Tolerance Compliance
