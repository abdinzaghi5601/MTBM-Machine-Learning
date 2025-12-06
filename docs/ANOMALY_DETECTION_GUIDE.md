# MTBM Deviation Anomaly Detection Guide

Complete guide to using machine learning for detecting anomalies in horizontal and vertical tunnel deviations.

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [How It Works](#how-it-works)
4. [ML Algorithms Used](#ml-algorithms-used)
5. [Understanding the Results](#understanding-the-results)
6. [Usage Examples](#usage-examples)
7. [Interpretation Guide](#interpretation-guide)
8. [Advanced Usage](#advanced-usage)

---

## Overview

### What is Anomaly Detection?

Anomaly detection identifies **unusual patterns** in your deviation data that differ from normal behavior. These anomalies might indicate:

- ⚠️ Equipment malfunctions
- ⚠️ Difficult ground conditions
- ⚠️ Steering system problems
- ⚠️ Survey equipment errors
- ⚠️ Operator issues

### Why Use ML for Anomaly Detection?

**Traditional approach:**
- Fixed thresholds (e.g., deviation > 50mm = bad)
- Misses subtle patterns
- Many false alarms
- Can't detect gradual deterioration

**ML approach:**
- Learns normal patterns from your data
- Detects subtle anomalies
- Fewer false positives
- Catches issues early

---

## Quick Start

### Option 1: Simple Analysis

```bash
cd MTBM-Machine-Learning

# Analyze your data with anomaly detection:
python analyze_with_anomalies.py --protocol AVN2400 --data ../data/raw/your_data.csv
```

### Option 2: Generate Test Data

```bash
# Test with synthetic data:
python analyze_with_anomalies.py --protocol AVN800 --generate-sample
```

### Option 3: High Sensitivity Detection

```bash
# More sensitive detection (catches more anomalies):
python analyze_with_anomalies.py --protocol AVN3000 --data your_data.csv --sensitivity high
```

### What You Get

After running, you'll find in `outputs/AVN2400/anomaly_detection/`:

1. **deviation_anomalies.png** - 6-panel visualization
2. **anomaly_report.txt** - Detailed text report
3. **anomaly_results.csv** - Full data with anomaly flags

---

## How It Works

### The Process

```
Your Data
    ↓
Feature Engineering (creates derived features)
    ↓
Train 5 ML Models (each learns normal patterns)
    ↓
Each model votes: Normal or Anomaly
    ↓
Ensemble Decision (majority vote)
    ↓
Anomaly Score (0-1, higher = more anomalous)
    ↓
Results + Visualizations
```

### Features Analyzed

The system analyzes **9 features** for each data point:

**Direct Measurements:**
1. Horizontal deviation (mm)
2. Vertical deviation (mm)
3. Total deviation (mm)

**Derived Features:**
4. Horizontal velocity (rate of change)
5. Vertical velocity (rate of change)
6. Horizontal acceleration (rate of change of velocity)
7. Vertical acceleration (rate of change of velocity)
8. Horizontal standard deviation (local variability)
9. Vertical standard deviation (local variability)

**Why these features?**
- Sudden changes in velocity → Steering problems
- High acceleration → Erratic control
- High local std dev → Unstable conditions

---

## ML Algorithms Used

The system uses **5 different ML algorithms** and combines their predictions:

### 1. Isolation Forest
**How it works:** Isolates anomalies by random partitioning
**Good at:** Finding points that are easy to separate from the rest
**Use case:** General-purpose anomaly detection

### 2. Local Outlier Factor (LOF)
**How it works:** Compares local density with neighbors
**Good at:** Finding points in low-density regions
**Use case:** Detecting deviations that don't follow the cluster

### 3. DBSCAN Clustering
**How it works:** Groups similar points, marks outliers
**Good at:** Finding points that don't belong to any cluster
**Use case:** Identifying completely different patterns

### 4. Statistical Z-Score
**How it works:** Measures standard deviations from mean
**Good at:** Finding statistical outliers
**Use case:** Detecting extreme values

### 5. Autoencoder Neural Network (Optional)
**How it works:** Neural network learns to compress/decompress data
**Good at:** Finding points that can't be reconstructed well
**Use case:** Complex non-linear anomalies

**Note:** Autoencoder requires TensorFlow and enough data (>100 samples)

### Ensemble Voting

**How decisions are made:**
- Each algorithm votes: Normal (1) or Anomaly (-1)
- Anomaly if **2 or more** algorithms agree
- Anomaly score = (votes for anomaly) / (total algorithms)

**Example:**
```
Isolation Forest: Anomaly
LOF: Anomaly
DBSCAN: Normal
Z-Score: Normal
Autoencoder: Anomaly

Result: 3/5 vote Anomaly → ANOMALY DETECTED
Anomaly Score: 3/5 = 0.6 (Medium-High)
```

---

## Understanding the Results

### The 6-Panel Visualization

#### Panel 1: Deviation Pattern with Anomalies (Top Left)
**Shows:** Scatter plot of horizontal vs vertical deviation
**Blue dots:** Normal operation
**Red X marks:** Detected anomalies

**What to look for:**
- ✅ **Anomalies scattered randomly:** Occasional issues, probably normal
- ⚠️ **Anomalies in a line:** Systematic bias, check calibration
- ⚠️ **Anomalies clustered:** Specific section has problems
- 🛑 **All anomalies far from center:** Severe systematic issues

#### Panel 2: Time Series with Anomalies (Top Right)
**Shows:** Total deviation over tunnel length/time
**Blue line:** Deviation trend
**Red X marks:** Anomalies

**What to look for:**
- ✅ **Anomalies at peaks:** ML correctly identifying high deviations
- ⚠️ **Anomalies in clusters:** Problem period
- ⚠️ **Anomalies during stable periods:** Might be false positives OR subtle issues
- 🛑 **Many consecutive anomalies:** Sustained problem

#### Panel 3: Anomaly Score Distribution (Middle Left)
**Shows:** Histogram of anomaly scores (0 to 1)
**Red line:** 0.5 threshold (anomaly if > 0.5)

**What to look for:**
- ✅ **Most scores near 0:** Mostly normal operation
- ✅ **Clear separation:** Good model performance
- ⚠️ **Many scores around 0.5:** Models disagree, uncertain
- ⚠️ **Bimodal (two peaks):** Two distinct operating modes

#### Panel 4: Model Agreement (Middle Right)
**Shows:** Correlation between ML models
**Warm colors (red):** Models agree
**Cool colors (blue):** Models disagree

**What to look for:**
- ✅ **High correlations (>0.7):** Models agree, reliable detection
- ⚠️ **Low correlations (<0.3):** Models see different patterns
- ℹ️ **Different algorithms:** It's normal to have some variation

#### Panels 5 & 6: Horizontal and Vertical Analysis (Bottom)
**Shows:** Individual deviation components with anomalies

**What to look for:**
- See if anomalies are in horizontal, vertical, or both
- Horizontal-only anomalies → Steering bias
- Vertical-only anomalies → Elevation control issues
- Both → General control problem

### The Text Report

**Section 1: Detection Summary**
- Total samples analyzed
- Number of anomalies found
- Percentage of data flagged

**Section 2: Severity Distribution**
- Low (anomaly score 0-0.3)
- Medium (anomaly score 0.3-0.6)
- High (anomaly score 0.6-1.0)

**Section 3: Model Detection Rates**
- How many anomalies each model found
- Shows if models agree

**Section 4: Top 10 Anomalies**
- Most anomalous points
- Location, deviations, score
- Use this to investigate worst cases

### The Results CSV

**New columns added:**
- `anomaly_isolation_forest`: 1 if this model detected anomaly
- `anomaly_lof`: LOF detection
- `anomaly_dbscan`: DBSCAN detection
- `anomaly_zscore`: Z-score detection
- `anomaly_autoencoder`: Autoencoder detection (if available)
- `anomaly_ensemble`: **Final decision** (1 = anomaly, 0 = normal)
- `anomaly_votes`: Number of models that detected anomaly
- `anomaly_score`: Anomaly probability (0-1)
- `anomaly_severity`: Low/Medium/High

**How to use:**
```python
import pandas as pd

# Load results
df = pd.read_csv('anomaly_results.csv')

# Filter only anomalies
anomalies = df[df['anomaly_ensemble'] == 1]

# High severity only
critical = df[df['anomaly_severity'] == 'High']

# Sort by score
worst_cases = df.sort_values('anomaly_score', ascending=False).head(20)
```

---

## Usage Examples

### Example 1: Basic Analysis

```bash
python analyze_with_anomalies.py \
    --protocol AVN2400 \
    --data tunnel_section_1.csv
```

**Output:**
- Standard protocol analysis plots
- Anomaly detection visualization
- Anomaly report
- Results CSV

### Example 2: High Sensitivity (More Detections)

```bash
python analyze_with_anomalies.py \
    --protocol AVN3000 \
    --data tunnel_data.csv \
    --sensitivity high
```

**Use when:**
- You want to be extra cautious
- Investigating problems
- Quality-critical sections

**Trade-off:** More false positives

### Example 3: Low Sensitivity (Fewer False Alarms)

```bash
python analyze_with_anomalies.py \
    --protocol AVN800 \
    --data tunnel_data.csv \
    --sensitivity low
```

**Use when:**
- Normal operations
- Reducing alert fatigue
- Established tunnels with good history

**Trade-off:** Might miss subtle issues

### Example 4: Anomaly Detection Only

```bash
python analyze_with_anomalies.py \
    --protocol AVN2400 \
    --data tunnel_data.csv \
    --anomaly-only
```

**Use when:**
- You already have standard plots
- Just want anomaly analysis
- Batch processing

### Example 5: Python Script

```python
from deviation_anomaly_detector import DeviationAnomalyDetector
import pandas as pd

# Load your data
df = pd.read_csv('my_data.csv')

# Create detector
detector = DeviationAnomalyDetector(
    protocol_name='AVN2400',
    sensitivity='medium'
)

# Train on data
detector.fit(df)

# Detect anomalies
results = detector.predict(df)

# Visualize
detector.plot_anomalies(results, save_path='anomalies.png')

# Get anomalies only
anomalies = results[results['anomaly_ensemble'] == 1]
print(f"Found {len(anomalies)} anomalies")
```

### Example 6: Batch Processing

```python
from pathlib import Path
from deviation_anomaly_detector import DeviationAnomalyDetector
import pandas as pd

# Process multiple files
data_files = Path('data/').glob('*.csv')

for file in data_files:
    print(f"\nProcessing: {file.name}")

    df = pd.read_csv(file)

    detector = DeviationAnomalyDetector('AVN2400', 'medium')
    detector.fit(df)
    results = detector.predict(df)

    # Save results
    output = f"anomalies_{file.stem}.csv"
    results.to_csv(output, index=False)

    print(f"Anomalies: {results['anomaly_ensemble'].sum()}")
```

---

## Interpretation Guide

### Sensitivity Levels

| Sensitivity | Contamination | Use Case | Trade-off |
|-------------|---------------|----------|-----------|
| **Low** | 5% | Normal operations | May miss subtle issues |
| **Medium** | 10% | Standard monitoring | Balanced |
| **High** | 15% | Investigation, critical work | More false positives |

### Anomaly Score Interpretation

| Score Range | Severity | Meaning | Action |
|-------------|----------|---------|--------|
| 0.0 - 0.3 | Low | Borderline, questionable | Monitor |
| 0.3 - 0.6 | Medium | Likely anomaly | Investigate |
| 0.6 - 1.0 | High | Very likely anomaly | Immediate action |

### Common Patterns

#### Pattern 1: Isolated Spikes
**What you see:** Few scattered anomalies
**Meaning:** Occasional unusual readings
**Action:** Review context, probably normal variation

#### Pattern 2: Clustered Anomalies
**What you see:** Anomalies grouped in time/space
**Meaning:** Specific period/location had issues
**Action:** Investigate that section (geology change? equipment issue?)

#### Pattern 3: Gradual Increase in Anomaly Score
**What you see:** Scores slowly rising over time
**Meaning:** Gradual deterioration
**Action:** Schedule maintenance, check wear

#### Pattern 4: High Anomaly Density
**What you see:** >20% of data flagged
**Meaning:** Either serious problems OR wrong sensitivity
**Action:** Check raw data, reduce sensitivity, or investigate thoroughly

#### Pattern 5: Model Disagreement
**What you see:** Some models detect, others don't
**Meaning:** Edge case, subtle pattern
**Action:** Manual review needed

### What Makes a Good vs Bad Result?

**Good Anomaly Detection:**
- ✅ <10% anomalies detected (medium sensitivity)
- ✅ Anomalies match high deviation periods
- ✅ Models mostly agree (high correlation)
- ✅ Clear separation in score distribution
- ✅ Anomalies make sense when reviewed

**Concerning Anomaly Detection:**
- ⚠️ >20% anomalies (too many, reduce sensitivity or investigate)
- ⚠️ Anomalies during stable periods (possible false positives)
- ⚠️ Models disagree (<0.5 correlation)
- ⚠️ Uniform score distribution (model confused)
- ⚠️ All high severity (serious problems or wrong threshold)

---

## Advanced Usage

### Customizing the Detector

```python
from deviation_anomaly_detector import DeviationAnomalyDetector

# Create custom detector
detector = DeviationAnomalyDetector('AVN3000', 'medium')

# Modify contamination manually
detector.contamination = 0.08  # 8% expected anomalies

# Change Z-score threshold
detector.z_threshold = 2.8  # More sensitive

# Train and predict
detector.fit(df)
results = detector.predict(df)
```

### Analyzing Specific Sections

```python
# Analyze just a problematic section
section = df[(df['tunnel_length_m'] >= 100) & (df['tunnel_length_m'] <= 200)]

detector = DeviationAnomalyDetector('AVN2400', 'high')
detector.fit(section)
results = detector.predict(section)
```

### Comparing Different Sensitivities

```python
for sensitivity in ['low', 'medium', 'high']:
    detector = DeviationAnomalyDetector('AVN2400', sensitivity)
    detector.fit(df)
    results = detector.predict(df)

    n_anomalies = results['anomaly_ensemble'].sum()
    print(f"{sensitivity:8s}: {n_anomalies} anomalies")
```

### Exporting Anomalies for Review

```python
# Get high-severity anomalies only
high_severity = results[results['anomaly_severity'] == 'High'].copy()

# Add context
high_severity['prev_deviation'] = results['total_deviation_mm'].shift(1)
high_severity['next_deviation'] = results['total_deviation_mm'].shift(-1)

# Sort by score
high_severity = high_severity.sort_values('anomaly_score', ascending=False)

# Export for review
high_severity.to_csv('critical_anomalies_for_review.csv', index=False)
```

---

## Troubleshooting

### Issue: Too Many Anomalies Detected

**Possible causes:**
- Sensitivity too high
- Data quality issues
- Actually have many problems

**Solutions:**
1. Try lower sensitivity: `--sensitivity low`
2. Check data quality (missing values, outliers)
3. Investigate if real problems exist

### Issue: No Anomalies Detected

**Possible causes:**
- Sensitivity too low
- Data very uniform
- Insufficient data

**Solutions:**
1. Try higher sensitivity: `--sensitivity high`
2. Check if data has variation
3. Need at least 100+ samples

### Issue: Models Disagree

**Possible causes:**
- Edge cases
- Complex patterns
- Insufficient training data

**Solutions:**
1. Review manually
2. Collect more data
3. Focus on high-score anomalies (ensemble agrees)

### Issue: TensorFlow Not Available

**Message:** "Autoencoder will be disabled"

**Solution:** Install TensorFlow (optional):
```bash
pip install tensorflow
```

**Or:** Ignore - 4 other algorithms still work fine

---

## Best Practices

### 1. Start with Medium Sensitivity
Begin with medium, then adjust based on results.

### 2. Review High-Severity Anomalies Manually
Don't blindly trust ML - always review critical cases.

### 3. Look for Patterns
Isolated anomalies less concerning than clusters.

### 4. Use Multiple Runs
Try different sensitivities, compare results.

### 5. Combine with Domain Knowledge
ML finds patterns, you interpret them.

### 6. Keep Historical Baselines
Train on known-good data, detect deviations from that.

### 7. Regular Retraining
Retrain periodically as conditions change.

### 8. Document Anomalies
Note what you found, create knowledge base.

---

## Summary

### What You Learned

✅ How to run anomaly detection on deviation data
✅ What the 5 ML algorithms do
✅ How to interpret the visualizations
✅ When to use different sensitivity levels
✅ How to identify real problems vs false positives

### Quick Reference

**Run basic analysis:**
```bash
python analyze_with_anomalies.py --protocol AVN2400 --data mydata.csv
```

**High sensitivity:**
```bash
python analyze_with_anomalies.py --protocol AVN3000 --data mydata.csv --sensitivity high
```

**Check outputs:**
```
outputs/AVN2400/anomaly_detection/
├── deviation_anomalies_*.png  ← Visualizations
├── anomaly_report_*.txt       ← Text report
└── anomaly_results_*.csv      ← Data with flags
```

**Key columns in results:**
- `anomaly_ensemble`: 1 = anomaly, 0 = normal
- `anomaly_score`: 0-1, higher = more anomalous
- `anomaly_severity`: Low/Medium/High

---

**Document Version:** 1.0
**Last Updated:** November 2024
**For:** MTBM Deviation Anomaly Detection System
