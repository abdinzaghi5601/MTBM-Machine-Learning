# Anomaly Detection Quick Start

## 🚀 In 30 Seconds

```bash
cd MTBM-Machine-Learning

# Detect anomalies in your deviation data:
python analyze_with_anomalies.py --protocol AVN2400 --data ../data/raw/yourfile.csv

# Or test with sample data:
python analyze_with_anomalies.py --protocol AVN800 --generate-sample
```

**Done!** Results in `outputs/AVN2400/anomaly_detection/`

---

## 🎯 What It Does

Uses **5 machine learning algorithms** to find unusual patterns in:
- ✅ Horizontal deviation
- ✅ Vertical deviation
- ✅ Rate of change (velocity)
- ✅ Acceleration
- ✅ Local variability

**Detects:**
- Equipment malfunctions
- Difficult ground conditions
- Steering problems
- Survey errors
- Gradual deterioration

---

## 📊 What You Get

### 1. Visualization (`deviation_anomalies.png`)
6-panel plot showing:
- Deviation pattern with anomalies marked
- Time series with red X on anomalies
- Anomaly score distribution
- Model agreement analysis
- Horizontal/vertical breakdown

### 2. Report (`anomaly_report.txt`)
- Detection summary
- Severity breakdown
- Top 10 worst anomalies
- Model performance

### 3. Results CSV (`anomaly_results.csv`)
Original data + new columns:
- `anomaly_ensemble`: 1 = anomaly, 0 = normal
- `anomaly_score`: 0-1 (higher = more anomalous)
- `anomaly_severity`: Low/Medium/High
- Individual model predictions

---

## 🎚️ Sensitivity Levels

```bash
# Low sensitivity (fewer false alarms, ~5% flagged)
python analyze_with_anomalies.py --protocol AVN2400 --data mydata.csv --sensitivity low

# Medium sensitivity (balanced, ~10% flagged) - DEFAULT
python analyze_with_anomalies.py --protocol AVN2400 --data mydata.csv

# High sensitivity (catch everything, ~15% flagged)
python analyze_with_anomalies.py --protocol AVN2400 --data mydata.csv --sensitivity high
```

**Use high sensitivity when:**
- Investigating known problems
- Quality-critical sections
- Want to be extra cautious

**Use low sensitivity when:**
- Normal operations
- Reducing alert fatigue
- Good track record

---

## 🤖 The 5 ML Algorithms

| Algorithm | What It Detects |
|-----------|-----------------|
| **Isolation Forest** | Easy-to-separate outliers |
| **Local Outlier Factor** | Points in low-density regions |
| **DBSCAN** | Points that don't cluster |
| **Z-Score** | Statistical outliers |
| **Autoencoder** | Complex non-linear patterns |

**Ensemble voting:** Anomaly if **2 or more** algorithms agree

---

## 📈 Interpreting Results

### Anomaly Score

| Score | Severity | Meaning | Action |
|-------|----------|---------|--------|
| 0.0-0.3 | Low | Borderline | Monitor |
| 0.3-0.6 | Medium | Likely anomaly | Investigate |
| 0.6-1.0 | High | Very anomalous | Immediate action |

### Good vs Concerning

**Good result:**
- ✅ <10% anomalies detected
- ✅ Anomalies match high deviation periods
- ✅ Models mostly agree
- ✅ Makes sense when reviewed

**Concerning result:**
- ⚠️ >20% anomalies (too many)
- ⚠️ Anomalies during stable periods
- ⚠️ Models disagree
- ⚠️ Doesn't match your expectations

---

## 💡 Common Commands

### Basic Analysis
```bash
python analyze_with_anomalies.py --protocol AVN2400 --data mydata.csv
```

### High Sensitivity
```bash
python analyze_with_anomalies.py --protocol AVN3000 --data mydata.csv --sensitivity high
```

### Anomaly Detection Only (Skip Standard Plots)
```bash
python analyze_with_anomalies.py --protocol AVN2400 --data mydata.csv --anomaly-only
```

### Generate Test Data
```bash
python analyze_with_anomalies.py --protocol AVN800 --generate-sample --samples 2000
```

---

## 🔍 Where Are My Results?

```
outputs/
└── AVN2400/                    ← Your protocol
    └── anomaly_detection/      ← Anomaly results here
        ├── deviation_anomalies_*.png    ← Visualization
        ├── anomaly_report_*.txt         ← Text report
        └── anomaly_results_*.csv        ← Data with flags
```

---

## 🐍 Python API

```python
from deviation_anomaly_detector import DeviationAnomalyDetector
import pandas as pd

# Load data
df = pd.read_csv('my_data.csv')

# Create detector
detector = DeviationAnomalyDetector(
    protocol_name='AVN2400',
    sensitivity='medium'
)

# Train
detector.fit(df)

# Detect anomalies
results = detector.predict(df)

# Get anomalies only
anomalies = results[results['anomaly_ensemble'] == 1]
print(f"Found {len(anomalies)} anomalies")

# Visualize
detector.plot_anomalies(results, save_path='anomalies.png')

# Generate report
detector.generate_anomaly_report(results, save_path='report.txt')
```

---

## ❓ Quick Troubleshooting

### Too many anomalies?
```bash
# Try lower sensitivity
python analyze_with_anomalies.py --protocol AVN2400 --data mydata.csv --sensitivity low
```

### No anomalies found?
```bash
# Try higher sensitivity
python analyze_with_anomalies.py --protocol AVN2400 --data mydata.csv --sensitivity high
```

### "TensorFlow not available" warning?
**Ignore it** - 4 other algorithms still work fine
**Or install:** `pip install tensorflow`

### Models disagree?
**Normal** - review high-score anomalies manually

---

## 📖 Learn More

- **Complete Guide:** `docs/ANOMALY_DETECTION_GUIDE.md` (detailed)
- **Plot Interpretation:** `docs/PLOT_INTERPRETATION_GUIDE.md`
- **Multi-Protocol:** `docs/MULTI_PROTOCOL_GUIDE.md`

---

## 🎯 Best Practices

1. ✅ Start with medium sensitivity
2. ✅ Review high-severity anomalies manually
3. ✅ Look for patterns (clusters vs isolated)
4. ✅ Compare different sensitivities
5. ✅ Combine ML results with your expertise

---

## 📊 Example Output Summary

```
ANOMALY DETECTION SUMMARY
==================================================
Total Samples: 1,000
Anomalies Detected: 87 (8.7%)
High Severity: 12 (1.2%)
Sensitivity Level: medium

By Model:
  isolation_forest    : 76 (7.6%)
  lof                 : 91 (9.1%)
  dbscan              : 63 (6.3%)
  zscore              : 45 (4.5%)
  autoencoder         : 82 (8.2%)

By Severity:
  High   : 12 (1.2%)
  Medium : 38 (3.8%)
  Low    : 37 (3.7%)
```

---

**You're ready to detect anomalies!** 🚀

**Created:** November 2024
**Version:** 1.0
