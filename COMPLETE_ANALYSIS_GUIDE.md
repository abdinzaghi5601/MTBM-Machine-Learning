# Complete MTBM Analysis Guide

## 🎯 Everything in One Command

```bash
cd MTBM-Machine-Learning

# Complete analysis with all features:
python full_analysis.py --protocol AVN2400 --data yourfile.csv --diameter 800
```

This runs:
✅ Standard protocol analysis
✅ ML anomaly detection
✅ Pipe bore tolerance compliance
✅ Integrated reporting

---

## 📊 What You Get

### 1. Standard Protocol Analysis
- Time series plots
- Deviation analysis
- Performance dashboard
- Correlation matrix

### 2. ML Anomaly Detection
- 5 ML algorithms detecting unusual patterns
- Anomaly visualizations with red X markers
- Severity classification (Low/Medium/High)
- Detailed anomaly report

### 3. Pipe Bore Tolerance Compliance
- Industry-standard tolerance checking
- Quality ratings (Excellent/Good/Acceptable/Marginal/Poor)
- Compliance reports
- Tolerance visualization

### 4. Integrated Summary
- Combined analysis across all modules
- Critical findings identification
- Complete results dataset

---

## 🔧 Pipe Bore Tolerances

Based on industry standards:

| Pipe Diameter | Vertical Tolerance | Horizontal Tolerance |
|---------------|-------------------|---------------------|
| < 600 mm | ± 20 mm | ± 25 mm |
| 600-1000 mm | ± 25 mm | ± 40 mm |
| 1000-1400 mm | ± 30 mm | ± 100 mm |
| > 1400 mm | ± 50 mm | ± 200 mm |

**System automatically applies correct tolerances based on your pipe diameter!**

---

## 💡 Common Commands

### Full Analysis (All Features)
```bash
python full_analysis.py --protocol AVN2400 --data tunnel.csv --diameter 800
```

### With High Sensitivity Anomaly Detection
```bash
python full_analysis.py --protocol AVN3000 --data tunnel.csv --diameter 1200 --sensitivity high
```

### Skip Anomaly Detection (Faster)
```bash
python full_analysis.py --protocol AVN2400 --data tunnel.csv --diameter 600 --skip-anomaly
```

### Skip Tolerance Check
```bash
python full_analysis.py --protocol AVN2400 --data tunnel.csv --diameter 800 --skip-tolerance
```

### Generate Test Data
```bash
python full_analysis.py --protocol AVN2400 --generate-sample --diameter 800
```

---

## 📁 Output Structure

```
outputs/
└── AVN2400/
    └── complete_analysis/
        ├── anomaly_detection/
        │   ├── anomalies_*.png
        │   ├── anomaly_report_*.txt
        │   └── anomaly_results_*.csv
        ├── tolerance_compliance/
        │   ├── compliance_*.png
        │   ├── compliance_report_*.txt
        │   └── compliance_results_*.csv
        ├── integrated_summary_*.txt  ← READ THIS FIRST
        └── complete_results_*.csv    ← Full data with all flags
```

---

## 📈 Understanding the Results

### Integrated Summary File

**Section 1: Deviation Statistics**
- Mean, std, max for horizontal, vertical, total deviation

**Section 2: Anomaly Detection Summary**
- Number of anomalies
- Severity breakdown

**Section 3: Tolerance Compliance Summary**
- Compliance rate
- Quality distribution

**Section 4: Critical Findings**
- Automatic identification of issues:
  - ⚠️ High anomaly rate (>20%)
  - ⚠️ Low compliance rate (<80%)
  - ⚠️ Excessive high-severity anomalies
  - ✅ No critical issues (all good!)

### Complete Results CSV

**Original columns** + **New columns:**

**From Anomaly Detection:**
- `anomaly_ensemble`: 1 = anomaly, 0 = normal
- `anomaly_score`: 0-1 (higher = more anomalous)
- `anomaly_severity`: Low/Medium/High
- Individual model predictions

**From Tolerance Compliance:**
- `hor_tolerance_mm`: Applicable horizontal tolerance
- `vert_tolerance_mm`: Applicable vertical tolerance
- `both_within_tolerance`: TRUE/FALSE
- `quality_rating`: Excellent/Good/Acceptable/Marginal/Poor
- `hor_exceedance_pct`: % of horizontal tolerance used
- `vert_exceedance_pct`: % of vertical tolerance used

---

## 🎯 Quality Standards

### Anomaly Score
| Score | Severity | Action |
|-------|----------|--------|
| 0.0-0.3 | Low | Monitor |
| 0.3-0.6 | Medium | Investigate |
| 0.6-1.0 | High | Immediate action |

### Tolerance Exceedance
| Exceedance | Quality | Meaning |
|------------|---------|---------|
| ≤ 50% | Excellent | Well within limits |
| 50-80% | Good | Acceptable performance |
| 80-100% | Acceptable | At tolerance limit |
| 100-120% | Marginal | Slightly over |
| > 120% | Poor | Significantly over |

---

## 🚦 Decision Guide

### Excellent Project (All Good ✅)
- <5% anomalies detected
- >95% tolerance compliance
- Quality rating mostly "Excellent" or "Good"
- No critical findings

### Good Project (Minor Issues ⚠️)
- 5-10% anomalies
- 85-95% tolerance compliance
- Quality rating mostly "Good" or "Acceptable"
- Few critical findings

### Concerning Project (Needs Attention 🛑)
- >15% anomalies
- <80% tolerance compliance
- Quality rating "Marginal" or "Poor"
- Multiple critical findings

**Actions:**
- Review anomaly reports
- Check tolerance compliance reports
- Investigate specific problem areas
- Consider operational adjustments

---

## 🐍 Python API Example

```python
from deviation_anomaly_detector import DeviationAnomalyDetector
from pipe_bore_tolerances import PipeBoreToleranceSystem
import pandas as pd

# Load data
df = pd.read_csv('tunnel_data.csv')

# 1. Anomaly Detection
detector = DeviationAnomalyDetector('AVN2400', 'medium')
detector.fit(df)
df_with_anomalies = detector.predict(df)

# 2. Tolerance Compliance
tolerance_system = PipeBoreToleranceSystem()
df_with_compliance = tolerance_system.assess_dataframe(
    df_with_anomalies,
    diameter_mm=800
)

# 3. Get critical cases
critical = df_with_compliance[
    (df_with_compliance['anomaly_severity'] == 'High') &
    (~df_with_compliance['both_within_tolerance'])
]

print(f"Critical cases requiring attention: {len(critical)}")
```

---

## ❓ FAQ

### Q: What diameter should I use?
**A:** The actual bore diameter of your pipe in millimeters (e.g., 800 for 800mm pipe)

### Q: Which sensitivity should I choose?
**A:**
- **Low** - Normal operations, reduce false alarms
- **Medium** - Standard monitoring (default)
- **High** - Investigation, critical sections

### Q: What if I don't know my pipe diameter?
**A:** Check your project specifications. For testing, use common sizes: 600, 800, 1000, 1200

### Q: Can I skip parts of the analysis?
**A:** Yes! Use `--skip-anomaly` or `--skip-tolerance` flags

### Q: How long does it take?
**A:**
- Standard analysis: ~10-30 seconds
- With anomaly detection: +20-60 seconds
- Total: Usually under 2 minutes

---

## 📊 Example Output

```
COMPLETE ANALYSIS FINISHED!
========================================================

Outputs saved to: outputs/AVN2400/complete_analysis/

Generated Files:
  📊 Standard Analysis: outputs/AVN2400/plots
  🔍 Anomaly Detection: outputs/AVN2400/complete_analysis/anomaly_detection
  📏 Tolerance Compliance: outputs/AVN2400/complete_analysis/tolerance_compliance
  📄 Integrated Summary: outputs/AVN2400/complete_analysis/integrated_summary_*.txt
  📁 Complete Results: outputs/AVN2400/complete_analysis/complete_results_*.csv

QUICK STATISTICS
========================================================
  Total Samples: 1,000
  Max Horizontal Deviation: 23.4 mm
  Max Vertical Deviation: 18.9 mm
  Anomalies: 87 (8.7%)
  Tolerance Compliance: 962/1000 (96.2%)
```

---

## 🎓 Best Practices

1. **Start with complete analysis** to see everything
2. **Review integrated summary first** for overview
3. **Check critical findings** section
4. **Investigate high-severity anomalies** manually
5. **Review tolerance violations** in detail
6. **Use appropriate sensitivity** for your situation
7. **Document findings** for future reference

---

## 🚀 You're Ready!

Run your first complete analysis:

```bash
python full_analysis.py --protocol AVN2400 --data your_data.csv --diameter 800
```

Then open:
1. `integrated_summary_*.txt` - Start here!
2. Anomaly visualizations - See detected anomalies
3. Tolerance compliance plots - Check quality
4. `complete_results_*.csv` - Full data for further analysis

**Everything you need for comprehensive MTBM tunnel deviation analysis!** ✅

---

**Created:** November 2024
**Version:** 1.0
**Includes:** Protocol Analysis + ML Anomaly Detection + Tolerance Compliance
