# Multi-Protocol Quick Start

## 🚀 In 30 Seconds

```bash
cd MTBM-Machine-Learning

# Analyze YOUR data with YOUR protocol:
python analyze_protocol.py --protocol AVN2400 --data ../data/raw/yourfile.csv

# Or generate sample data to test:
python analyze_protocol.py --protocol AVN800 --generate-sample
```

**Done!** Results are in `outputs/AVN2400/` (or whichever protocol you chose).

---

## 🎯 Choose Your Protocol

| Protocol | Use When You Have | Parameters |
|----------|-------------------|------------|
| **AVN800** | Basic MTBM system | 15+ core measurements |
| **AVN1200** | + Yaw/Pitch sensors | 18+ with orientation |
| **AVN2400** | + Drill head sensors | 22+ with forces |
| **AVN3000** | Full advanced system | 23+ complete monitoring |

**Not sure?** Try AVN2400 - it works for most cases.

---

## 📁 Where Are My Results?

After running, find your outputs here:

```
outputs/
└── AVN2400/              ← Your protocol name
    ├── plots/            ← All PNG visualizations
    ├── data/             ← Processed CSV data
    └── reports/          ← Text analysis reports
```

Each protocol gets its own folder - no conflicts!

---

## 💡 Common Commands

### Analyze Your CSV Data
```bash
python analyze_protocol.py --protocol AVN2400 --data ../data/raw/mydata.csv
```

### Generate Test Data
```bash
python analyze_protocol.py --protocol AVN800 --generate-sample
```

### Check Data Compatibility
```bash
python analyze_protocol.py --protocol AVN3000 --data mydata.csv --validate-only
```

### Fast Analysis (No Plots)
```bash
python analyze_protocol.py --protocol AVN1200 --data mydata.csv --no-plots
```

### Custom Sample Size
```bash
python analyze_protocol.py --protocol AVN2400 --generate-sample --samples 5000
```

---

## 🔍 What You Get

Every analysis produces:

**4 Visualizations:**
1. `mtbm_time_series_overview.png` - All parameters over time
2. `mtbm_deviation_analysis.png` - Tunnel alignment quality
3. `mtbm_performance_dashboard.png` - Operational efficiency
4. `mtbm_correlation_matrix.png` - Parameter relationships

**Data File:**
- Processed CSV with timestamp

**Text Report:**
- Protocol-specific analysis
- Quality metrics
- Parameter statistics

---

## ❓ Quick Troubleshooting

### "ModuleNotFoundError: protocol_configs"
**Fix:** Make sure you're in the right directory:
```bash
cd MTBM-Machine-Learning
python analyze_protocol.py ...
```

### "Unknown protocol: AVN 2400"
**Fix:** No spaces in protocol name:
```bash
python analyze_protocol.py --protocol AVN2400  # ✅ Correct
```

### "earth_pressure values outside range"
**Fix:** Your earth pressure might need scaling (divide by 10):
```python
df['earth_pressure_01_bar'] = df['earth_pressure_01_bar'] / 10
```

### Missing parameters warning
**Normal:** Lower protocols don't have all parameters. Analysis continues with available data.

---

## 🎓 Learn More

- **Full Multi-Protocol Guide:** `docs/MULTI_PROTOCOL_GUIDE.md` (25 pages)
- **Understanding Plots:** `docs/PLOT_INTERPRETATION_GUIDE.md`
- **Code Details:** `docs/CODE_STRUCTURE_GUIDE.md`
- **General Usage:** `QUICK_START.md`

---

## 📊 Example Output

After running:
```bash
python analyze_protocol.py --protocol AVN2400 --generate-sample
```

You'll see:
```
Initialized AVN 2400 Analyzer
Outputs will be saved to: /path/to/outputs/AVN2400

Generating 1000 sample records for AVN 2400
Generated data with 22 parameters

Validating data against AVN 2400 specifications...
Validation complete:
  Errors: 0
  Warnings: 0

======================================================================
Running AVN 2400 Analysis
======================================================================

✅ Saved processed data: /path/to/outputs/AVN2400/data/...

📊 Generating visualizations...
  Creating time series overview...
  Saved: /path/to/outputs/AVN2400/plots/mtbm_time_series_overview.png

  Creating deviation analysis...
  Saved: /path/to/outputs/AVN2400/plots/mtbm_deviation_analysis.png

  Creating performance dashboard...
  Saved: /path/to/outputs/AVN2400/plots/mtbm_performance_dashboard.png

  Creating correlation matrix...
  Saved: /path/to/outputs/AVN2400/plots/mtbm_correlation_matrix.png

📄 Generating analysis report...
✅ Saved analysis report: /path/to/outputs/AVN2400/reports/...

======================================================================
✅ Analysis Complete!
======================================================================

Outputs saved to:
  📊 Plots: /path/to/outputs/AVN2400/plots
  📁 Data: /path/to/outputs/AVN2400/data
  📄 Reports: /path/to/outputs/AVN2400/reports
```

Then open the plots from `outputs/AVN2400/plots/` and review!

---

## 🔄 Comparing Protocols

Want to see differences between protocols?

```bash
# Run same data through different protocols:
python analyze_protocol.py --protocol AVN1200 --data mydata.csv
python analyze_protocol.py --protocol AVN2400 --data mydata.csv

# Compare:
# outputs/AVN1200/plots/ vs outputs/AVN2400/plots/
```

See how AVN2400's extra parameters provide more insight!

---

## ✅ You're Ready!

That's it! You can now:
- ✅ Analyze data from any AVN protocol
- ✅ Generate test data for any protocol
- ✅ Get protocol-specific visualizations
- ✅ Compare different protocols
- ✅ Validate data quality

**Next step:** Try it with your own data! 🚀

---

**Created:** November 2024
**Version:** 1.0
**Protocols:** AVN 800, 1200, 2400, 3000
